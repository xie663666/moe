from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import MoEClassifier, sync_time
from .routing_profile import FixedExpertProfile


@dataclass
class EpochStats:
    loss: float
    acc: float
    step_ms: float
    router_ms: float
    dispatch_ms: float
    expert_forward_ms: float
    combine_ms: float
    expert_ms: float
    moe_ms: float
    router_ratio: float
    expert_ratio: float

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        d = {
            f"{prefix}loss": self.loss,
            f"{prefix}acc": self.acc,
            f"{prefix}step_ms": self.step_ms,
            f"{prefix}router_ms": self.router_ms,
            f"{prefix}dispatch_ms": self.dispatch_ms,
            f"{prefix}expert_forward_ms": self.expert_forward_ms,
            f"{prefix}combine_ms": self.combine_ms,
            f"{prefix}expert_ms": self.expert_ms,
            f"{prefix}moe_ms": self.moe_ms,
            f"{prefix}router_ratio": self.router_ratio,
            f"{prefix}expert_ratio": self.expert_ratio,
        }
        return d


@dataclass
class FitResult:
    best_epoch: int
    best_val_acc: float
    best_test_acc: float
    train_stats: Dict
    val_stats: Dict
    test_stats: Dict
    best_checkpoint_path: str


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


class CSVLogger:
    def __init__(self, filepath: str, fieldnames: Iterable[str]):
        self.filepath = filepath
        self.fieldnames = list(fieldnames)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._initialized = os.path.exists(filepath) and os.path.getsize(filepath) > 0

    def log(self, row: Dict):
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow(row)


RESULT_FIELDNAMES = [
    "pair_name",
    "direction",
    "source_group",
    "target_group",
    "num_experts",
    "top_k",
    "num_fixed",
    "fixed_mass_mode",
    "freeze_fixed_experts",
    "source_checkpoint",
    "profile_path",
    "transfer_run_dir",
    "best_epoch",
    "best_val_acc",
    "best_test_acc",
    "train_loss",
    "train_acc",
    "train_step_ms",
    "train_router_ms",
    "train_dispatch_ms",
    "train_expert_forward_ms",
    "train_combine_ms",
    "train_expert_ms",
    "train_moe_ms",
    "train_router_ratio",
    "train_expert_ratio",
    "val_loss",
    "val_acc",
    "val_step_ms",
    "val_router_ms",
    "val_dispatch_ms",
    "val_expert_forward_ms",
    "val_combine_ms",
    "val_expert_ms",
    "val_moe_ms",
    "val_router_ratio",
    "val_expert_ratio",
    "test_loss",
    "test_acc",
    "test_step_ms",
    "test_router_ms",
    "test_dispatch_ms",
    "test_expert_forward_ms",
    "test_combine_ms",
    "test_expert_ms",
    "test_moe_ms",
    "test_router_ratio",
    "test_expert_ratio",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(params, lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, epochs: int):
    return CosineAnnealingLR(optimizer, T_max=max(epochs, 1))


def save_checkpoint(path: str, model: nn.Module, optimizer, scheduler, epoch: int, extra: Optional[Dict] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "extra": extra or {},
    }
    torch.save(payload, path)


@torch.no_grad()
def evaluate_one_epoch(
    model: MoEClassifier,
    loader: DataLoader,
    device: torch.device,
    fixed_profile: Optional[FixedExpertProfile] = None,
    routing_mode: str = "normal",
    timing_interval: int = 1,
) -> EpochStats:
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    step_ms_meter = AverageMeter()
    router_ms_meter = AverageMeter()
    dispatch_ms_meter = AverageMeter()
    expert_forward_ms_meter = AverageMeter()
    combine_ms_meter = AverageMeter()
    expert_ms_meter = AverageMeter()
    moe_ms_meter = AverageMeter()

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        timing_enabled = (step % max(1, timing_interval) == 0)

        step_start = sync_time() if timing_enabled else None
        outputs = model(images, routing_mode=routing_mode, fixed_profile=fixed_profile, timing_enabled=timing_enabled)
        logits = outputs["logits"]
        aux_loss = outputs["route_info"]["aux_loss"]
        loss = F.cross_entropy(logits, targets) + aux_loss
        if timing_enabled:
            step_end = sync_time()
            step_ms_meter.update((step_end - step_start) * 1000.0)
            timing = outputs["route_info"]["timing"]
            router_ms_meter.update(timing["router_ms"])
            dispatch_ms_meter.update(timing["dispatch_ms"])
            expert_forward_ms_meter.update(timing["expert_forward_ms"])
            combine_ms_meter.update(timing["combine_ms"])
            expert_ms_meter.update(timing["expert_ms"])
            moe_ms_meter.update(timing["moe_ms"])

        preds = torch.argmax(logits, dim=1)
        batch_acc = (preds == targets).float().mean().item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(batch_acc, images.size(0))

    router_ratio = router_ms_meter.avg / max(step_ms_meter.avg, 1e-12)
    expert_ratio = expert_ms_meter.avg / max(step_ms_meter.avg, 1e-12)
    return EpochStats(
        loss=loss_meter.avg,
        acc=acc_meter.avg,
        step_ms=step_ms_meter.avg,
        router_ms=router_ms_meter.avg,
        dispatch_ms=dispatch_ms_meter.avg,
        expert_forward_ms=expert_forward_ms_meter.avg,
        combine_ms=combine_ms_meter.avg,
        expert_ms=expert_ms_meter.avg,
        moe_ms=moe_ms_meter.avg,
        router_ratio=router_ratio,
        expert_ratio=expert_ratio,
    )


def train_one_epoch(
    model: MoEClassifier,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    fixed_profile: Optional[FixedExpertProfile] = None,
    routing_mode: str = "normal",
    timing_interval: int = 20,
    grad_clip: float = 0.0,
    progress_desc: str = "train",
) -> EpochStats:
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    step_ms_meter = AverageMeter()
    router_ms_meter = AverageMeter()
    dispatch_ms_meter = AverageMeter()
    expert_forward_ms_meter = AverageMeter()
    combine_ms_meter = AverageMeter()
    expert_ms_meter = AverageMeter()
    moe_ms_meter = AverageMeter()

    pbar = tqdm(loader, desc=progress_desc, leave=False)
    for step, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        timing_enabled = (step % max(1, timing_interval) == 0)

        optimizer.zero_grad(set_to_none=True)
        step_start = sync_time() if timing_enabled else None
        outputs = model(images, routing_mode=routing_mode, fixed_profile=fixed_profile, timing_enabled=timing_enabled)
        logits = outputs["logits"]
        aux_loss = outputs["route_info"]["aux_loss"]
        loss = F.cross_entropy(logits, targets) + aux_loss
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        if timing_enabled:
            step_end = sync_time()
            step_ms_meter.update((step_end - step_start) * 1000.0)
            timing = outputs["route_info"]["timing"]
            router_ms_meter.update(timing["router_ms"])
            dispatch_ms_meter.update(timing["dispatch_ms"])
            expert_forward_ms_meter.update(timing["expert_forward_ms"])
            combine_ms_meter.update(timing["combine_ms"])
            expert_ms_meter.update(timing["expert_ms"])
            moe_ms_meter.update(timing["moe_ms"])

        preds = torch.argmax(logits, dim=1)
        batch_acc = (preds == targets).float().mean().item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(batch_acc, images.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    router_ratio = router_ms_meter.avg / max(step_ms_meter.avg, 1e-12)
    expert_ratio = expert_ms_meter.avg / max(step_ms_meter.avg, 1e-12)
    return EpochStats(
        loss=loss_meter.avg,
        acc=acc_meter.avg,
        step_ms=step_ms_meter.avg,
        router_ms=router_ms_meter.avg,
        dispatch_ms=dispatch_ms_meter.avg,
        expert_forward_ms=expert_forward_ms_meter.avg,
        combine_ms=combine_ms_meter.avg,
        expert_ms=expert_ms_meter.avg,
        moe_ms=moe_ms_meter.avg,
        router_ratio=router_ratio,
        expert_ratio=expert_ratio,
    )


def fit_model(
    model: MoEClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    run_dir: str,
    fixed_profile: Optional[FixedExpertProfile] = None,
    routing_mode: str = "normal",
    timing_interval: int = 20,
    grad_clip: float = 0.0,
) -> FitResult:
    os.makedirs(run_dir, exist_ok=True)
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, epochs=epochs)

    best_val_acc = -math.inf
    best_epoch = -1
    best_checkpoint_path = os.path.join(run_dir, "best.pt")
    last_checkpoint_path = os.path.join(run_dir, "last.pt")

    best_train_stats = None
    best_val_stats = None
    best_test_stats = None

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            fixed_profile=fixed_profile,
            routing_mode=routing_mode,
            timing_interval=timing_interval,
            grad_clip=grad_clip,
            progress_desc=f"train epoch {epoch}/{epochs}",
        )
        val_stats = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            fixed_profile=fixed_profile,
            routing_mode=routing_mode,
            timing_interval=1,
        )
        test_stats = evaluate_one_epoch(
            model=model,
            loader=test_loader,
            device=device,
            fixed_profile=fixed_profile,
            routing_mode=routing_mode,
            timing_interval=1,
        )
        scheduler.step()

        summary = {
            "epoch": epoch,
            **train_stats.to_dict(prefix="train_"),
            **val_stats.to_dict(prefix="val_"),
            **test_stats.to_dict(prefix="test_"),
            "lr": scheduler.get_last_lr()[0],
        }
        with open(os.path.join(run_dir, f"epoch_{epoch:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if val_stats.acc > best_val_acc:
            best_val_acc = val_stats.acc
            best_epoch = epoch
            best_train_stats = train_stats
            best_val_stats = val_stats
            best_test_stats = test_stats
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                extra={"train": train_stats.to_dict(), "val": val_stats.to_dict(), "test": test_stats.to_dict()},
            )

        save_checkpoint(last_checkpoint_path, model, optimizer, scheduler, epoch)

    return FitResult(
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        best_test_acc=best_test_stats.acc if best_test_stats is not None else float("nan"),
        train_stats=best_train_stats.to_dict(prefix="train_") if best_train_stats is not None else {},
        val_stats=best_val_stats.to_dict(prefix="val_") if best_val_stats is not None else {},
        test_stats=best_test_stats.to_dict(prefix="test_") if best_test_stats is not None else {},
        best_checkpoint_path=best_checkpoint_path,
    )
