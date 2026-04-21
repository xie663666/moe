from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import torch


@dataclass
class FixedExpertProfile:
    num_experts: int
    top_k: int
    num_fixed: int
    fixed_indices: torch.Tensor
    fixed_weights: torch.Tensor
    dynamic_indices: torch.Tensor
    fixed_mass: float
    source_group: str = ""
    target_group: str = ""
    source_checkpoint: str = ""

    def to_device(self, device: torch.device | str) -> "FixedExpertProfile":
        return FixedExpertProfile(
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_fixed=self.num_fixed,
            fixed_indices=self.fixed_indices.to(device),
            fixed_weights=self.fixed_weights.to(device),
            dynamic_indices=self.dynamic_indices.to(device),
            fixed_mass=self.fixed_mass,
            source_group=self.source_group,
            target_group=self.target_group,
            source_checkpoint=self.source_checkpoint,
        )

    def to_serializable_dict(self) -> Dict:
        data = asdict(self)
        data["fixed_indices"] = self.fixed_indices.tolist()
        data["fixed_weights"] = self.fixed_weights.tolist()
        data["dynamic_indices"] = self.dynamic_indices.tolist()
        return data

    @staticmethod
    def from_dict(data: Dict) -> "FixedExpertProfile":
        return FixedExpertProfile(
            num_experts=int(data["num_experts"]),
            top_k=int(data["top_k"]),
            num_fixed=int(data["num_fixed"]),
            fixed_indices=torch.tensor(data["fixed_indices"], dtype=torch.long),
            fixed_weights=torch.tensor(data["fixed_weights"], dtype=torch.float32),
            dynamic_indices=torch.tensor(data["dynamic_indices"], dtype=torch.long),
            fixed_mass=float(data["fixed_mass"]),
            source_group=data.get("source_group", ""),
            target_group=data.get("target_group", ""),
            source_checkpoint=data.get("source_checkpoint", ""),
        )


def build_fixed_expert_profile(
    avg_gate: torch.Tensor,
    top_k: int,
    num_fixed: int,
    source_group: str = "",
    target_group: str = "",
    source_checkpoint: str = "",
    mass_mode: str = "count_ratio",
) -> FixedExpertProfile:
    num_experts = int(avg_gate.numel())
    if not 0 <= num_fixed <= top_k <= num_experts:
        raise ValueError(f"Require 0 <= num_fixed <= top_k <= num_experts, got {num_fixed}, {top_k}, {num_experts}.")

    avg_gate = avg_gate.float().clone()
    avg_gate = avg_gate / avg_gate.sum().clamp_min(1e-12)

    if num_fixed == 0:
        return FixedExpertProfile(
            num_experts=num_experts,
            top_k=top_k,
            num_fixed=0,
            fixed_indices=torch.empty(0, dtype=torch.long),
            fixed_weights=torch.empty(0, dtype=torch.float32),
            dynamic_indices=torch.arange(num_experts, dtype=torch.long),
            fixed_mass=0.0,
            source_group=source_group,
            target_group=target_group,
            source_checkpoint=source_checkpoint,
        )

    fixed_indices = torch.topk(avg_gate, k=num_fixed, dim=0).indices.cpu()
    fixed_scores = avg_gate[fixed_indices].cpu()
    fixed_scores = fixed_scores / fixed_scores.sum().clamp_min(1e-12)

    if mass_mode == "count_ratio":
        fixed_mass = float(num_fixed) / float(top_k)
    elif mass_mode == "raw_profile":
        fixed_mass = float(avg_gate[fixed_indices].sum().item())
        if num_fixed == top_k:
            fixed_mass = 1.0
    else:
        raise ValueError(f"Unknown mass_mode: {mass_mode}")

    fixed_weights = fixed_scores * fixed_mass

    all_indices = torch.arange(num_experts, dtype=torch.long)
    mask = torch.ones(num_experts, dtype=torch.bool)
    mask[fixed_indices] = False
    dynamic_indices = all_indices[mask]

    return FixedExpertProfile(
        num_experts=num_experts,
        top_k=top_k,
        num_fixed=num_fixed,
        fixed_indices=fixed_indices,
        fixed_weights=fixed_weights.float(),
        dynamic_indices=dynamic_indices,
        fixed_mass=fixed_mass,
        source_group=source_group,
        target_group=target_group,
        source_checkpoint=source_checkpoint,
    )


@torch.no_grad()
def extract_average_gate_profile(model, dataloader, device: torch.device) -> torch.Tensor:
    model.eval()
    gate_sum: Optional[torch.Tensor] = None
    total_samples = 0

    for images, _targets in dataloader:
        images = images.to(device, non_blocking=True)
        outputs = model(images, routing_mode="normal", fixed_profile=None, timing_enabled=False)
        sparse_gates = outputs["route_info"]["sparse_gates"].detach()

        if gate_sum is None:
            gate_sum = torch.zeros(sparse_gates.size(1), device=device, dtype=sparse_gates.dtype)

        gate_sum += sparse_gates.sum(dim=0)
        total_samples += images.size(0)

    if gate_sum is None:
        raise RuntimeError("Cannot extract average gate profile from an empty dataloader.")

    avg_gate = gate_sum / max(total_samples, 1)
    avg_gate = avg_gate / avg_gate.sum().clamp_min(1e-12)
    return avg_gate.detach().cpu()
