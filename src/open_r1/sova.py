"""Bidirectional strict-oracle virtual advantages for SOVA-TW-GRPO."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


SOVA_MODES = frozenset({"positive", "negative", "bidirectional"})
SOVA_NUM_GENERATIONS = 8
SOVA_POSITIVE_VIRTUAL_TOTAL_REWARD = 1.5
SOVA_NEGATIVE_VIRTUAL_TOTAL_REWARD = 2.0


@dataclass(frozen=True)
class SOVAResult:
    """Advantages and branch diagnostics for the real policy samples only."""

    advantages: torch.Tensor
    positive_active_groups: torch.Tensor
    negative_active_groups: torch.Tensor
    positive_effective_mean: torch.Tensor
    positive_effective_std: torch.Tensor
    negative_effective_mean: torch.Tensor
    negative_effective_std: torch.Tensor
    positive_residual: torch.Tensor
    negative_residual: torch.Tensor


def validate_sova_configuration(
    mode: str,
    lambda_positive: float,
    lambda_negative: float,
    positive_virtual_total_reward: float = SOVA_POSITIVE_VIRTUAL_TOTAL_REWARD,
    negative_virtual_total_reward: float = SOVA_NEGATIVE_VIRTUAL_TOTAL_REWARD,
) -> None:
    """Fail closed on ambiguous or directionally invalid SOVA settings."""
    if mode not in SOVA_MODES:
        allowed = ", ".join(sorted(SOVA_MODES))
        raise ValueError(f"sova_mode must be one of: {allowed}")

    for name, value in (
        ("sova_lambda_positive", lambda_positive),
        ("sova_lambda_negative", lambda_negative),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and lie in [0, 1]")

    positive_enabled = mode in {"positive", "bidirectional"}
    negative_enabled = mode in {"negative", "bidirectional"}
    if positive_enabled != (lambda_positive > 0.0):
        raise ValueError(
            "sova_lambda_positive must be positive exactly when the positive branch is enabled"
        )
    if negative_enabled != (lambda_negative > 0.0):
        raise ValueError(
            "sova_lambda_negative must be positive exactly when the negative branch is enabled"
        )

    if (
        not math.isfinite(positive_virtual_total_reward)
        or not 0.0 <= positive_virtual_total_reward < 2.0
    ):
        raise ValueError(
            "sova_positive_virtual_total_reward must be finite and lie in [0, 2)"
        )
    if (
        not math.isfinite(negative_virtual_total_reward)
        or not 1.0 < negative_virtual_total_reward <= 2.0
    ):
        raise ValueError(
            "sova_negative_virtual_total_reward must be finite and lie in (1, 2]"
        )


def _validate_reward_vectors(
    baseline_advantages: torch.Tensor,
    total_rewards: torch.Tensor,
    accuracy_rewards: torch.Tensor,
    format_rewards: torch.Tensor,
    num_generations: int,
) -> None:
    tensors = {
        "baseline_advantages": baseline_advantages,
        "total_rewards": total_rewards,
        "accuracy_rewards": accuracy_rewards,
        "format_rewards": format_rewards,
    }
    if any(tensor.ndim != 1 for tensor in tensors.values()):
        raise ValueError("all SOVA tensors must be one-dimensional")
    if len({tensor.numel() for tensor in tensors.values()}) != 1:
        raise ValueError("all SOVA tensors must have the same length")
    if num_generations != SOVA_NUM_GENERATIONS:
        raise ValueError(
            f"SOVA is calibrated for num_generations={SOVA_NUM_GENERATIONS}"
        )
    if total_rewards.numel() % num_generations != 0:
        raise ValueError("num_generations must divide the reward count")
    if not torch.isfinite(torch.stack(tuple(tensors.values()))).all().item():
        raise ValueError("SOVA tensors must contain only finite values")
    if not (
        accuracy_rewards.ge(0).all()
        & accuracy_rewards.le(1).all()
        & format_rewards.ge(0).all()
        & format_rewards.le(1).all()
    ).item():
        raise ValueError("accuracy and format rewards must lie in [0, 1]")
    if not torch.equal(total_rewards, accuracy_rewards + format_rewards):
        raise ValueError("total_rewards must equal accuracy_rewards + format_rewards")


def _virtual_group_statistics(
    grouped_rewards: torch.Tensor,
    baseline_mean: torch.Tensor,
    baseline_std: torch.Tensor,
    active_groups: torch.Tensor,
    virtual_total_reward: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    virtual_rewards = grouped_rewards.new_full(
        (grouped_rewards.size(0), 1), virtual_total_reward
    )
    augmented_rewards = torch.cat((grouped_rewards, virtual_rewards), dim=1)
    virtual_mean = augmented_rewards.mean(dim=1)
    virtual_std = augmented_rewards.std(dim=1)
    return (
        torch.where(active_groups, virtual_mean, baseline_mean),
        torch.where(active_groups, virtual_std, baseline_std),
    )


def apply_sova_advantages(
    baseline_advantages: torch.Tensor,
    total_rewards: torch.Tensor,
    accuracy_rewards: torch.Tensor,
    format_rewards: torch.Tensor,
    num_generations: int,
    mode: str,
    lambda_positive: float,
    lambda_negative: float,
    positive_virtual_total_reward: float = SOVA_POSITIVE_VIRTUAL_TOTAL_REWARD,
    negative_virtual_total_reward: float = SOVA_NEGATIVE_VIRTUAL_TOTAL_REWARD,
    eps: float = 1e-4,
) -> SOVAResult:
    """Apply selected virtual-statistics residuals without adding policy samples.

    The negative branch reproduces the original SOVA gate: every real
    accuracy reward is zero.  The positive branch is stricter: every real
    accuracy and format reward must equal one.  Virtual rewards participate
    only in group mean/std; the returned advantage length remains ``B * G``.
    """
    validate_sova_configuration(
        mode,
        lambda_positive,
        lambda_negative,
        positive_virtual_total_reward,
        negative_virtual_total_reward,
    )
    _validate_reward_vectors(
        baseline_advantages,
        total_rewards,
        accuracy_rewards,
        format_rewards,
        num_generations,
    )
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a finite positive number")

    grouped_rewards = total_rewards.view(-1, num_generations)
    grouped_accuracy = accuracy_rewards.view(-1, num_generations)
    grouped_format = format_rewards.view(-1, num_generations)
    baseline_mean = grouped_rewards.mean(dim=1)
    baseline_std = grouped_rewards.std(dim=1)

    positive_active_groups = grouped_accuracy.eq(1).all(dim=1) & grouped_format.eq(1).all(
        dim=1
    )
    negative_active_groups = grouped_accuracy.eq(0).all(dim=1)
    if mode == "positive":
        negative_active_groups = torch.zeros_like(negative_active_groups)
    elif mode == "negative":
        positive_active_groups = torch.zeros_like(positive_active_groups)

    if torch.any(positive_active_groups & negative_active_groups):
        raise RuntimeError("positive and negative SOVA gates must be mutually exclusive")

    positive_mean, positive_std = _virtual_group_statistics(
        grouped_rewards,
        baseline_mean,
        baseline_std,
        positive_active_groups,
        positive_virtual_total_reward,
    )
    negative_mean, negative_std = _virtual_group_statistics(
        grouped_rewards,
        baseline_mean,
        baseline_std,
        negative_active_groups,
        negative_virtual_total_reward,
    )

    expanded_positive_mean = positive_mean.repeat_interleave(num_generations)
    expanded_positive_std = positive_std.repeat_interleave(num_generations)
    expanded_negative_mean = negative_mean.repeat_interleave(num_generations)
    expanded_negative_std = negative_std.repeat_interleave(num_generations)
    expanded_positive_active = positive_active_groups.repeat_interleave(num_generations)
    expanded_negative_active = negative_active_groups.repeat_interleave(num_generations)

    full_positive_advantages = (total_rewards - expanded_positive_mean) / (
        expanded_positive_std + eps
    )
    full_negative_advantages = (total_rewards - expanded_negative_mean) / (
        expanded_negative_std + eps
    )
    positive_residual = torch.where(
        expanded_positive_active,
        full_positive_advantages - baseline_advantages,
        torch.zeros_like(baseline_advantages),
    )
    negative_residual = torch.where(
        expanded_negative_active,
        full_negative_advantages - baseline_advantages,
        torch.zeros_like(baseline_advantages),
    )

    affected = expanded_positive_active | expanded_negative_active
    candidate_advantages = (
        baseline_advantages
        + lambda_positive * positive_residual
        + lambda_negative * negative_residual
    )
    advantages = torch.where(affected, candidate_advantages, baseline_advantages)
    return SOVAResult(
        advantages=advantages,
        positive_active_groups=positive_active_groups,
        negative_active_groups=negative_active_groups,
        positive_effective_mean=positive_mean,
        positive_effective_std=positive_std,
        negative_effective_mean=negative_mean,
        negative_effective_std=negative_std,
        positive_residual=positive_residual,
        negative_residual=negative_residual,
    )
