from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, List, Sequence, Tuple

import torch

from src.cifar100_groups import PAIR_NAME_MAP, SIMILAR_GROUP_PAIRS, canonical_group_name
from src.datasets import get_group_datasets
from src.model import MoEClassifier, freeze_fixed_experts, load_source_trunk_weights
from src.routing_profile import build_fixed_expert_profile, extract_average_gate_profile
from src.train_utils import CSVLogger, RESULT_FIELDNAMES, fit_model, make_loader, set_seed


def parse_int_list(text: str) -> List[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError(f"Cannot parse any integers from: {text}")
    return values


def parse_pair_list(text: str) -> List[Tuple[str, str]]:
    text = text.strip().lower()
    if text == "all":
        return list(SIMILAR_GROUP_PAIRS)

    pairs = []
    for raw_pair in text.split(","):
        raw_pair = raw_pair.strip()
        if not raw_pair:
            continue
        if "->" in raw_pair:
            a, b = raw_pair.split("->", 1)
        elif ":" in raw_pair:
            a, b = raw_pair.split(":", 1)
        elif "|" in raw_pair:
            a, b = raw_pair.split("|", 1)
        else:
            raise ValueError(
                "Pair format must be 'group_a->group_b'. Example: aquatic_mammals->fish,flowers->trees"
            )
        pairs.append((canonical_group_name(a), canonical_group_name(b)))
    return pairs


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_model(args, num_classes: int) -> MoEClassifier:
    return MoEClassifier(
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        expert_hidden_dim=args.expert_hidden_dim,
        num_experts=args.current_num_experts,
        top_k=args.current_top_k,
        router_temperature=args.router_temperature,
        load_balance_coef=args.load_balance_coef,
        dropout=args.dropout,
    )


def ensure_run_dirs(base_dir: str, pair_name: str, direction: str, num_experts: int, top_k: int):
    run_dir = os.path.join(base_dir, pair_name, direction, f"E{num_experts}_K{top_k}")
    os.makedirs(run_dir, exist_ok=True)
    return {
        "root": run_dir,
        "source": os.path.join(run_dir, "source_train"),
        "profiles": os.path.join(run_dir, "profiles"),
        "transfer": os.path.join(run_dir, "transfer_runs"),
    }


def save_profile(avg_gate: torch.Tensor, filepath: str, meta: dict):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    payload = {"avg_gate": avg_gate.cpu(), "meta": meta}
    torch.save(payload, filepath)


def maybe_load_profile(filepath: str):
    payload = torch.load(filepath, map_location="cpu", weights_only=True)
    return payload["avg_gate"], payload.get("meta", {})


def run_direction(
    source_group: str,
    target_group: str,
    args,
    device: torch.device,
    csv_logger: CSVLogger,
):
    pair_name = PAIR_NAME_MAP.get((source_group, target_group), f"{source_group}_to_{target_group}")
    direction = f"{source_group}_to_{target_group}"
    run_dirs = ensure_run_dirs(args.output_dir, pair_name, direction, args.current_num_experts, args.current_top_k)

    source_train_ds, source_val_ds, source_test_ds, source_info = get_group_datasets(
        root=args.data_root,
        group_name=source_group,
        val_ratio=args.val_ratio,
        seed=args.seed,
        download=not args.no_download,
    )
    target_train_ds, target_val_ds, target_test_ds, target_info = get_group_datasets(
        root=args.data_root,
        group_name=target_group,
        val_ratio=args.val_ratio,
        seed=args.seed,
        download=not args.no_download,
    )

    source_train_loader = make_loader(
        source_train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    source_profile_loader = make_loader(
        source_train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    source_val_loader = make_loader(
        source_val_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    source_test_loader = make_loader(
        source_test_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    target_train_loader = make_loader(
        target_train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    target_val_loader = make_loader(
        target_val_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    target_test_loader = make_loader(
        target_test_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    source_ckpt_path = os.path.join(run_dirs["source"], "best.pt")
    profile_path = os.path.join(run_dirs["profiles"], "avg_gate.pt")

    if args.resume_source and os.path.exists(source_ckpt_path) and os.path.exists(profile_path):
        print(f"[Reuse source] {source_ckpt_path}")
        avg_gate, _meta = maybe_load_profile(profile_path)
    else:
        print(
            f"\n=== Source training: {source_group} | E={args.current_num_experts}, K={args.current_top_k} ==="
        )
        source_model = build_model(args, num_classes=source_info.num_classes).to(device)
        source_fit = fit_model(
            model=source_model,
            train_loader=source_train_loader,
            val_loader=source_val_loader,
            test_loader=source_test_loader,
            device=device,
            epochs=args.source_epochs,
            lr=args.source_lr,
            weight_decay=args.weight_decay,
            run_dir=run_dirs["source"],
            fixed_profile=None,
            routing_mode="normal",
            timing_interval=args.timing_interval,
            grad_clip=args.grad_clip,
        )
        print(
            f"[Source done] {source_group} best_epoch={source_fit.best_epoch} "
            f"best_val_acc={source_fit.best_val_acc:.4f} best_test_acc={source_fit.best_test_acc:.4f}"
        )

        source_model.load_state_dict(
            torch.load(source_fit.best_checkpoint_path, map_location=device, weights_only=True)["model_state"]
        )
        avg_gate = extract_average_gate_profile(source_model, source_profile_loader, device=device)
        save_profile(
            avg_gate,
            profile_path,
            meta={
                "source_group": source_group,
                "target_group": target_group,
                "num_experts": args.current_num_experts,
                "top_k": args.current_top_k,
                "source_checkpoint": source_fit.best_checkpoint_path,
                "avg_gate_sum": float(avg_gate.sum().item()),
                "avg_gate": avg_gate.tolist(),
            },
        )
        source_ckpt_path = source_fit.best_checkpoint_path

    for num_fixed in range(args.current_top_k + 1):
        fixed_profile = build_fixed_expert_profile(
            avg_gate=avg_gate,
            top_k=args.current_top_k,
            num_fixed=num_fixed,
            source_group=source_group,
            target_group=target_group,
            source_checkpoint=source_ckpt_path,
            mass_mode=args.fixed_mass_mode,
        )

        transfer_run_dir = os.path.join(run_dirs["transfer"], f"fixed_{num_fixed}")
        print(
            f"\n=== Transfer: {source_group} -> {target_group} | E={args.current_num_experts}, "
            f"K={args.current_top_k}, fixed={num_fixed} ==="
        )
        model = build_model(args, num_classes=target_info.num_classes).to(device)
        load_source_trunk_weights(model, source_ckpt_path, device=device)
        if args.freeze_fixed_experts:
            freeze_fixed_experts(model, fixed_profile)

        transfer_fit = fit_model(
            model=model,
            train_loader=target_train_loader,
            val_loader=target_val_loader,
            test_loader=target_test_loader,
            device=device,
            epochs=args.target_epochs,
            lr=args.target_lr,
            weight_decay=args.weight_decay,
            run_dir=transfer_run_dir,
            fixed_profile=fixed_profile,
            routing_mode="hybrid" if num_fixed > 0 else "normal",
            timing_interval=args.timing_interval,
            grad_clip=args.grad_clip,
        )
        print(
            f"[Transfer done] {source_group}->{target_group}, fixed={num_fixed}, "
            f"best_val_acc={transfer_fit.best_val_acc:.4f}, best_test_acc={transfer_fit.best_test_acc:.4f}"
        )

        row = {
            "pair_name": pair_name,
            "direction": direction,
            "source_group": source_group,
            "target_group": target_group,
            "num_experts": args.current_num_experts,
            "top_k": args.current_top_k,
            "num_fixed": num_fixed,
            "fixed_mass_mode": args.fixed_mass_mode,
            "freeze_fixed_experts": int(args.freeze_fixed_experts),
            "source_checkpoint": source_ckpt_path,
            "profile_path": profile_path,
            "transfer_run_dir": transfer_run_dir,
            "best_epoch": transfer_fit.best_epoch,
            "best_val_acc": transfer_fit.best_val_acc,
            "best_test_acc": transfer_fit.best_test_acc,
        }
        row.update(transfer_fit.train_stats)
        row.update(transfer_fit.val_stats)
        row.update(transfer_fit.test_stats)
        csv_logger.log(row)

        summary_path = os.path.join(transfer_run_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="MoE expert-transfer experiments on CIFAR-100 similar superclass pairs.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--pairs", type=str, default="all")
    parser.add_argument("--num_experts_list", type=str, default="8,16")
    parser.add_argument("--top_k_list", type=str, default="2,4")
    parser.add_argument("--source_epochs", type=int, default=30)
    parser.add_argument("--target_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--source_lr", type=float, default=1e-3)
    parser.add_argument("--target_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--expert_hidden_dim", type=int, default=512)
    parser.add_argument("--router_temperature", type=float, default=1.0)
    parser.add_argument("--load_balance_coef", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--fixed_mass_mode", type=str, default="count_ratio", choices=["count_ratio", "raw_profile"])
    parser.add_argument("--freeze_fixed_experts", action="store_true")
    parser.add_argument("--timing_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume_source", action="store_true")
    parser.add_argument("--no_download", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    selected_pairs = parse_pair_list(args.pairs)
    num_experts_list = parse_int_list(args.num_experts_list)
    top_k_list = parse_int_list(args.top_k_list)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_logger = CSVLogger(os.path.join(args.output_dir, "results.csv"), RESULT_FIELDNAMES)

    for num_experts in num_experts_list:
        for top_k in top_k_list:
            if top_k > num_experts:
                print(f"[Skip] top_k={top_k} > num_experts={num_experts}")
                continue
            args.current_num_experts = num_experts
            args.current_top_k = top_k

            for group_a, group_b in selected_pairs:
                run_direction(group_a, group_b, args, device, csv_logger)
                run_direction(group_b, group_a, args, device, csv_logger)


if __name__ == "__main__":
    main()
