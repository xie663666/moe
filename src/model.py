from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routing_profile import FixedExpertProfile


def sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallBackbone(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 64, stride=1),
            ConvBNAct(64, 64, stride=1),
            nn.MaxPool2d(2),
            ConvBNAct(64, 128, stride=1),
            ConvBNAct(128, 128, stride=1),
            nn.MaxPool2d(2),
            ConvBNAct(128, 256, stride=1),
            ConvBNAct(256, 256, stride=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        x = self.proj(x)
        x = self.norm(x)
        return x


class ExpertMLP(nn.Module):
    def __init__(self, hidden_dim: int, expert_hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, expert_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearRouter(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass
class RouteComputation:
    sparse_gates: torch.Tensor
    selected_idx: torch.Tensor
    selected_weights: torch.Tensor
    dynamic_probs: Optional[torch.Tensor]
    aux_loss: torch.Tensor


class MoELayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        expert_hidden_dim: int,
        num_experts: int,
        top_k: int,
        router_temperature: float = 1.0,
        load_balance_coef: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"Require 1 <= top_k <= num_experts, got {top_k}, {num_experts}.")
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_temperature = router_temperature
        self.load_balance_coef = load_balance_coef

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.router = LinearRouter(hidden_dim, num_experts)
        self.experts = nn.ModuleList(
            [ExpertMLP(hidden_dim, expert_hidden_dim, dropout=dropout) for _ in range(num_experts)]
        )

    def _build_sparse_gates(self, probs: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        top_vals, top_idx = torch.topk(probs, k=top_k, dim=-1)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sparse_gates = torch.zeros_like(probs)
        sparse_gates.scatter_(1, top_idx, top_vals)
        return sparse_gates, top_idx, top_vals

    def _load_balance_loss(self, probs: Optional[torch.Tensor], selected_idx: torch.Tensor) -> torch.Tensor:
        if self.load_balance_coef <= 0.0 or probs is None or probs.numel() == 0:
            return torch.tensor(0.0, device=selected_idx.device)

        num_available = probs.size(1)
        mean_prob = probs.mean(dim=0)

        one_hot = F.one_hot(selected_idx, num_classes=num_available).float()
        mean_topk = one_hot.mean(dim=(0, 1))

        aux = num_available * torch.sum(mean_prob * mean_topk)
        return aux * self.load_balance_coef

    def _route_normal(self, x: torch.Tensor) -> RouteComputation:
        logits = self.router(x)
        probs = F.softmax(logits / self.router_temperature, dim=-1)
        sparse_gates, selected_idx, selected_weights = self._build_sparse_gates(probs, self.top_k)
        aux_loss = self._load_balance_loss(probs, selected_idx)
        return RouteComputation(
            sparse_gates=sparse_gates,
            selected_idx=selected_idx,
            selected_weights=selected_weights,
            dynamic_probs=probs,
            aux_loss=aux_loss,
        )

    def _route_hybrid_fixed(self, x: torch.Tensor, fixed_profile: FixedExpertProfile) -> RouteComputation:
        device = x.device
        batch_size = x.size(0)
        num_fixed = fixed_profile.num_fixed

        if num_fixed == 0:
            return self._route_normal(x)

        fixed_profile = fixed_profile.to_device(device)
        sparse_gates = torch.zeros(batch_size, self.num_experts, device=device, dtype=x.dtype)

        fixed_idx = fixed_profile.fixed_indices
        fixed_weights = fixed_profile.fixed_weights.to(device=device, dtype=x.dtype)
        sparse_gates[:, fixed_idx] = fixed_weights.unsqueeze(0).expand(batch_size, -1)

        if num_fixed == self.top_k:
            selected_idx = fixed_idx.unsqueeze(0).expand(batch_size, -1)
            selected_weights = fixed_weights.unsqueeze(0).expand(batch_size, -1)
            return RouteComputation(
                sparse_gates=sparse_gates,
                selected_idx=selected_idx,
                selected_weights=selected_weights,
                dynamic_probs=None,
                aux_loss=torch.tensor(0.0, device=device),
            )

        dyn_idx = fixed_profile.dynamic_indices
        dyn_k = self.top_k - num_fixed
        dyn_mass = 1.0 - float(fixed_profile.fixed_mass)

        dyn_weight = self.router.linear.weight[dyn_idx]
        dyn_bias = None if self.router.linear.bias is None else self.router.linear.bias[dyn_idx]
        dyn_logits = F.linear(x, dyn_weight, dyn_bias)
        dyn_probs = F.softmax(dyn_logits / self.router_temperature, dim=-1)

        dyn_top_vals, dyn_top_local_idx = torch.topk(dyn_probs, k=dyn_k, dim=-1)
        dyn_top_vals = dyn_top_vals / dyn_top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        dyn_top_vals = dyn_top_vals * dyn_mass
        dyn_top_global_idx = dyn_idx[dyn_top_local_idx]
        sparse_gates.scatter_(1, dyn_top_global_idx, dyn_top_vals)

        selected_idx = torch.cat([fixed_idx.unsqueeze(0).expand(batch_size, -1), dyn_top_global_idx], dim=-1)
        selected_weights = torch.cat(
            [fixed_weights.unsqueeze(0).expand(batch_size, -1), dyn_top_vals],
            dim=-1,
        )
        aux_loss = self._load_balance_loss(dyn_probs, dyn_top_local_idx)

        return RouteComputation(
            sparse_gates=sparse_gates,
            selected_idx=selected_idx,
            selected_weights=selected_weights,
            dynamic_probs=dyn_probs,
            aux_loss=aux_loss,
        )

    def _dispatch_and_combine(
        self,
        x: torch.Tensor,
        sparse_gates: torch.Tensor,
        timing_enabled: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        output = torch.zeros_like(x)
        dispatch_ms = 0.0
        expert_forward_ms = 0.0
        combine_ms = 0.0

        for expert_idx, expert in enumerate(self.experts):
            dispatch_start = sync_time() if timing_enabled else None
            expert_gate = sparse_gates[:, expert_idx]
            active_mask = expert_gate > 0
            if not torch.any(active_mask):
                if timing_enabled:
                    dispatch_ms += (sync_time() - dispatch_start) * 1000.0
                continue

            expert_input = x[active_mask]
            if timing_enabled:
                dispatch_ms += (sync_time() - dispatch_start) * 1000.0

            expert_forward_start = sync_time() if timing_enabled else None
            expert_output = expert(expert_input)
            if timing_enabled:
                expert_forward_ms += (sync_time() - expert_forward_start) * 1000.0

            combine_start = sync_time() if timing_enabled else None
            output[active_mask] += expert_output * expert_gate[active_mask].unsqueeze(-1)
            if timing_enabled:
                combine_ms += (sync_time() - combine_start) * 1000.0

        if not timing_enabled:
            return output, None

        return output, {
            "dispatch_ms": dispatch_ms,
            "expert_forward_ms": expert_forward_ms,
            "combine_ms": combine_ms,
        }

    def forward(
        self,
        x: torch.Tensor,
        routing_mode: str = "normal",
        fixed_profile: Optional[FixedExpertProfile] = None,
        timing_enabled: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        residual = x
        x = self.input_norm(x)

        moe_start = sync_time() if timing_enabled else None
        router_start = sync_time() if timing_enabled else None

        if routing_mode == "hybrid" and fixed_profile is not None and fixed_profile.num_fixed > 0:
            route = self._route_hybrid_fixed(x, fixed_profile)
        else:
            route = self._route_normal(x)

        router_end = sync_time() if timing_enabled else None
        moe_out, expert_timing = self._dispatch_and_combine(
            x,
            route.sparse_gates,
            timing_enabled=timing_enabled,
        )
        combine_finalize_start = sync_time() if timing_enabled else None
        moe_out = self.output_norm(residual + moe_out)
        combine_finalize_end = sync_time() if timing_enabled else None

        timing = None
        if timing_enabled:
            dispatch_ms = expert_timing["dispatch_ms"]
            expert_forward_ms = expert_timing["expert_forward_ms"]
            combine_ms = expert_timing["combine_ms"] + (combine_finalize_end - combine_finalize_start) * 1000.0
            expert_ms = dispatch_ms + expert_forward_ms + combine_ms
            timing = {
                "router_ms": (router_end - router_start) * 1000.0,
                "dispatch_ms": dispatch_ms,
                "expert_forward_ms": expert_forward_ms,
                "combine_ms": combine_ms,
                "expert_ms": expert_ms,
                "moe_ms": (combine_finalize_end - moe_start) * 1000.0,
            }

        route_info = {
            "sparse_gates": route.sparse_gates,
            "selected_idx": route.selected_idx,
            "selected_weights": route.selected_weights,
            "aux_loss": route.aux_loss,
            "timing": timing,
        }
        return moe_out, route_info


class MoEClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        expert_hidden_dim: int = 512,
        num_experts: int = 8,
        top_k: int = 2,
        router_temperature: float = 1.0,
        load_balance_coef: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = SmallBackbone(hidden_dim=hidden_dim)
        self.moe = MoELayer(
            hidden_dim=hidden_dim,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            router_temperature=router_temperature,
            load_balance_coef=load_balance_coef,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        images: torch.Tensor,
        routing_mode: str = "normal",
        fixed_profile: Optional[FixedExpertProfile] = None,
        timing_enabled: bool = False,
    ) -> Dict:
        features = self.backbone(images)
        moe_features, route_info = self.moe(
            features,
            routing_mode=routing_mode,
            fixed_profile=fixed_profile,
            timing_enabled=timing_enabled,
        )
        logits = self.head(moe_features)
        return {
            "logits": logits,
            "features": moe_features,
            "route_info": route_info,
        }


def freeze_fixed_experts(model: MoEClassifier, fixed_profile: Optional[FixedExpertProfile]):
    if fixed_profile is None or fixed_profile.num_fixed == 0:
        return

    fixed_set = set(int(idx) for idx in fixed_profile.fixed_indices.tolist())
    for expert_idx, expert in enumerate(model.moe.experts):
        trainable = expert_idx not in fixed_set
        for param in expert.parameters():
            param.requires_grad = trainable


def load_source_trunk_weights(
    model: MoEClassifier,
    checkpoint_path: str,
    device: torch.device,
    strict_backbone_and_moe: bool = True,
) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state"]

    trunk_state = {
        key: value
        for key, value in state_dict.items()
        if key.startswith("backbone.") or key.startswith("moe.")
    }
    missing, unexpected = model.load_state_dict(trunk_state, strict=False)

    if strict_backbone_and_moe:
        unexpected_non_head = [name for name in unexpected if not name.startswith("head.")]
        missing_non_head = [name for name in missing if not name.startswith("head.")]
        if unexpected_non_head or missing_non_head:
            raise RuntimeError(
                f"Trunk weight loading mismatch. Missing={missing_non_head}, Unexpected={unexpected_non_head}"
            )

    return checkpoint
