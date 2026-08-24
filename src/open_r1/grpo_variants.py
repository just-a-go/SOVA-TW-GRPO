"""NGRPO and AVSPO advantage transforms for the GRPO loss."""

from __future__ import annotations

from collections.abc import Iterator, Sized
from dataclasses import dataclass
import math
from typing import Callable, Optional

import torch
from torch.utils.data import Sampler


NGRPO_MAX_TOTAL_REWARD = 1.0
NGRPO_STD_EPS = 1e-6
NGRPO_EPSILON_NEGATIVE = 0.16
NGRPO_EPSILON_POSITIVE = 0.24
NGRPO_NUM_ITERATIONS = 2

AVSPO_INITIAL_THRESHOLD = 0.5
AVSPO_SENSITIVITY = 0.5
AVSPO_THRESHOLD_LR = 0.01
AVSPO_COLLAPSE_THRESHOLD = 1e-6
AVSPO_THRESHOLD_MIN = 0.1
AVSPO_THRESHOLD_MAX = 0.9
AVSPO_ANCHOR_REWARD = 0.1
AVSPO_STD_EPS = 1e-4
AVSPO_CLIP_EPSILON = 0.2


@dataclass(frozen=True)
class NGRPOResult:
    advantages: torch.Tensor
    effective_mean: torch.Tensor
    effective_std: torch.Tensor
    all_correct_groups: torch.Tensor


@dataclass(frozen=True)
class NGRPOFilteredLoss:
    loss: torch.Tensor
    global_kept_sequences: int

    @property
    def skip_update(self) -> bool:
        return self.global_kept_sequences == 0


@dataclass
class NGRPOStepGate:
    entry_global_step: Optional[int] = None
    rewound: bool = False

    @property
    def armed(self) -> bool:
        return self.entry_global_step is not None

    def arm(self, global_step: int) -> None:
        if self.armed:
            raise RuntimeError("NGRPO skip gate is already armed")
        self.entry_global_step = int(global_step)
        self.rewound = False

    def before_counter_increment(self, state) -> bool:
        if not self.armed:
            return False
        if self.rewound or state.global_step != self.entry_global_step:
            raise RuntimeError("NGRPO skip gate reached an invalid pre-step state")
        state.global_step -= 1
        self.rewound = True
        return True

    def after_counter_increment(self, state, control) -> bool:
        if not self.armed:
            return False
        if not self.rewound or state.global_step != self.entry_global_step:
            raise RuntimeError("NGRPO skip gate reached an invalid post-step state")
        control.should_log = False
        control.should_save = False
        control.should_evaluate = False
        self.entry_global_step = None
        self.rewound = False
        return True


class RepeatGlobalBatchSampler(Sampler[int]):
    """Repeat complete global batches so every rank reuses its own rollout."""

    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        repeat_count: int,
        seed: int,
    ) -> None:
        size = len(data_source)
        if batch_size < 1 or repeat_count < 1 or size < batch_size:
            raise ValueError("invalid repeated-batch sampler configuration")
        self.size = size
        self.batch_size = int(batch_size)
        self.repeat_count = int(repeat_count)
        self.seed = int(seed)
        self.epoch = 0
        self.usable_size = size // batch_size * batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator)[
            : self.usable_size
        ].tolist()
        for start in range(0, self.usable_size, self.batch_size):
            batch = indices[start : start + self.batch_size]
            for _ in range(self.repeat_count):
                yield from batch

    def __len__(self) -> int:
        return self.usable_size * self.repeat_count


def ngrpo_backward_or_arm(
    backward_fn: Callable[..., None],
    loss: torch.Tensor,
    skip_update: bool,
    gate: NGRPOStepGate,
    global_step: int,
    **kwargs,
) -> bool:
    if skip_update:
        gate.arm(global_step)
        return False
    backward_fn(loss, **kwargs)
    return True


@dataclass(frozen=True)
class AVSPOResult:
    advantages: torch.Tensor
    collapsed_groups: torch.Tensor
    active_groups: torch.Tensor
    virtual_count: int
    virtual_rewards: torch.Tensor
    effective_mean: torch.Tensor
    effective_std: torch.Tensor


@dataclass(frozen=True)
class AVSPOState:
    threshold: float = AVSPO_INITIAL_THRESHOLD
    previous_mean_reward: Optional[float] = None


def _validate_rewards(rewards: torch.Tensor, num_generations: int) -> None:
    if not isinstance(rewards, torch.Tensor) or rewards.ndim != 1:
        raise ValueError("rewards must be a one-dimensional tensor")
    if not rewards.is_floating_point():
        raise ValueError("rewards must use a floating-point dtype")
    if (
        rewards.numel() == 0
        or num_generations < 2
        or rewards.numel() % num_generations != 0
    ):
        raise ValueError(
            "rewards must be non-empty and divisible by num_generations >= 2"
        )
    if not torch.isfinite(rewards).all().item():
        raise ValueError("rewards must contain only finite values")


def compute_ngrpo_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    max_reward: float = NGRPO_MAX_TOTAL_REWARD,
    eps: float = NGRPO_STD_EPS,
) -> NGRPOResult:
    """Append one virtual task-maximum reward to every group's statistics."""
    _validate_rewards(rewards, num_generations)
    if not math.isfinite(max_reward) or not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("max_reward must be finite and eps must be positive")
    if torch.any(rewards > max_reward).item():
        raise ValueError("max_reward must bound every observed reward")

    grouped = rewards.view(-1, num_generations)
    virtual = grouped.new_full((grouped.size(0), 1), max_reward)
    augmented = torch.cat((grouped, virtual), dim=1)
    mean = augmented.mean(dim=1)
    # Match the released NGRPO code and this repository's sample std.
    std = augmented.std(dim=1)
    advantages = (grouped - mean.unsqueeze(1)) / (std.unsqueeze(1) + eps)
    all_correct = grouped.eq(max_reward).all(dim=1)
    return NGRPOResult(advantages.flatten(), mean, std, all_correct)


def ngrpo_filtered_sequence_loss(
    sequence_loss: torch.Tensor,
    all_correct_groups: torch.Tensor,
    num_generations: int,
    num_processes: int,
    reduce_fn: Callable[..., torch.Tensor],
) -> NGRPOFilteredLoss:
    if sequence_loss.ndim != 1 or all_correct_groups.ndim != 1:
        raise ValueError("NGRPO loss inputs must be one-dimensional")
    if all_correct_groups.dtype != torch.bool:
        raise ValueError("all_correct_groups must be boolean")
    if sequence_loss.numel() != all_correct_groups.numel() * num_generations:
        raise ValueError("NGRPO group and sequence counts do not match")
    if num_processes < 1:
        raise ValueError("num_processes must be positive")

    keep_bool = (~all_correct_groups).repeat_interleave(num_generations)
    keep = keep_bool.to(sequence_loss.dtype)
    global_keep = reduce_fn(keep_bool.sum(dtype=torch.int64), reduction="sum")
    global_kept_sequences = int(global_keep.item())
    if global_kept_sequences > 0:
        loss = sequence_loss.mul(keep).sum() * num_processes / global_keep
    else:
        loss = sequence_loss.sum() * 0.0
    return NGRPOFilteredLoss(loss, global_kept_sequences)


def compute_avspo_virtual_count(
    global_acr: float,
    num_generations: int,
    sensitivity: float = AVSPO_SENSITIVITY,
) -> int:
    if not math.isfinite(global_acr) or not 0.0 <= global_acr <= 1.0:
        raise ValueError("global_acr must be finite and lie in [0, 1]")
    if num_generations < 2:
        raise ValueError("num_generations must be at least two")
    if not math.isfinite(sensitivity) or not 0.0 < sensitivity <= 1.0:
        raise ValueError("sensitivity must be finite and lie in (0, 1]")
    return max(
        1,
        min(
            num_generations,
            math.ceil(num_generations * global_acr**sensitivity),
        ),
    )


def find_avspo_collapsed_groups(
    rewards: torch.Tensor,
    num_generations: int,
    collapse_threshold: float = AVSPO_COLLAPSE_THRESHOLD,
) -> torch.Tensor:
    _validate_rewards(rewards, num_generations)
    if not math.isfinite(collapse_threshold) or collapse_threshold <= 0.0:
        raise ValueError("collapse_threshold must be finite and positive")
    grouped = rewards.view(-1, num_generations)
    return grouped.std(dim=1, unbiased=False) < collapse_threshold


def reduce_avspo_batch_statistics(
    collapsed_groups: torch.Tensor,
    rewards: torch.Tensor,
    reduce_fn: Callable[..., torch.Tensor],
) -> tuple[float, float]:
    if collapsed_groups.ndim != 1 or collapsed_groups.dtype != torch.bool:
        raise ValueError("collapsed_groups must be a one-dimensional bool tensor")
    if collapsed_groups.numel() == 0:
        raise ValueError("collapsed_groups must be non-empty")
    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("rewards must be a non-empty one-dimensional tensor")

    local = torch.stack(
        (
            collapsed_groups.float().sum(),
            rewards.new_tensor(float(collapsed_groups.numel())),
            rewards.detach().sum(),
            rewards.new_tensor(float(rewards.numel())),
        )
    )
    global_stats = reduce_fn(local, reduction="sum")
    if global_stats.shape != (4,) or not torch.isfinite(global_stats).all().item():
        raise ValueError("reduced AVSPO statistics are invalid")
    if global_stats[1].item() <= 0.0 or global_stats[3].item() <= 0.0:
        raise ValueError("reduced AVSPO counts must be positive")
    return (
        (global_stats[0] / global_stats[1]).item(),
        (global_stats[2] / global_stats[3]).item(),
    )


def apply_avspo_advantages(
    baseline_advantages: torch.Tensor,
    rewards: torch.Tensor,
    num_generations: int,
    global_acr: float,
    adaptive_threshold: float,
    sensitivity: float = AVSPO_SENSITIVITY,
    collapse_threshold: float = AVSPO_COLLAPSE_THRESHOLD,
    anchor_reward: float = AVSPO_ANCHOR_REWARD,
    eps: float = AVSPO_STD_EPS,
) -> AVSPOResult:
    """Recompute real-sample advantages for ACR-gated collapsed groups."""
    _validate_rewards(rewards, num_generations)
    if baseline_advantages.ndim != 1 or baseline_advantages.shape != rewards.shape:
        raise ValueError("baseline_advantages must match rewards")
    if not torch.isfinite(baseline_advantages).all().item():
        raise ValueError("baseline_advantages must contain only finite values")
    if not torch.all(rewards.eq(0.0) | rewards.eq(1.0)).item():
        raise ValueError("AVSPO requires binary rewards in {0, 1}")
    if not math.isfinite(global_acr) or not 0.0 <= global_acr <= 1.0:
        raise ValueError("global_acr must be finite and lie in [0, 1]")
    if not math.isfinite(adaptive_threshold) or not 0.0 <= adaptive_threshold <= 1.0:
        raise ValueError("adaptive_threshold must be finite and lie in [0, 1]")
    if not math.isfinite(anchor_reward) or anchor_reward <= 0.0:
        raise ValueError("anchor_reward must be finite and positive")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")

    grouped = rewards.view(-1, num_generations)
    baseline_grouped = baseline_advantages.view_as(grouped)
    baseline_mean = grouped.mean(dim=1)
    baseline_std = grouped.std(dim=1)
    # ACR uses population std; inactive groups retain the repository baseline.
    collapsed = find_avspo_collapsed_groups(
        rewards, num_generations, collapse_threshold
    )
    triggered = global_acr > adaptive_threshold
    active = collapsed & triggered

    if not triggered:
        empty = grouped.new_empty((grouped.size(0), 0))
        return AVSPOResult(
            baseline_advantages,
            collapsed,
            active,
            0,
            empty,
            baseline_mean,
            baseline_std,
        )

    virtual_count = compute_avspo_virtual_count(
        global_acr, num_generations, sensitivity
    )
    k = torch.arange(
        1, virtual_count + 1, device=grouped.device, dtype=grouped.dtype
    )
    observed_max = grouped.max(dim=1).values.unsqueeze(1)
    positive_virtual = observed_max * (1.0 - k / (virtual_count + 1.0))
    zero_virtual = anchor_reward * (virtual_count - k + 1.0) / virtual_count
    virtual = torch.where(observed_max > 0.0, positive_virtual, zero_virtual)

    augmented = torch.cat((grouped, virtual), dim=1)
    augmented_mean = augmented.mean(dim=1)
    augmented_std = augmented.std(dim=1, unbiased=False)
    candidate = (grouped - augmented_mean.unsqueeze(1)) / (
        augmented_std.unsqueeze(1) + eps
    )
    advantages = torch.where(active.unsqueeze(1), candidate, baseline_grouped)
    effective_mean = torch.where(active, augmented_mean, baseline_mean)
    effective_std = torch.where(active, augmented_std, baseline_std)
    return AVSPOResult(
        advantages.flatten(),
        collapsed,
        active,
        virtual_count,
        virtual,
        effective_mean,
        effective_std,
    )


def advance_avspo_state(
    state: AVSPOState,
    global_acr: float,
    mean_reward: float,
    learning_rate: float = AVSPO_THRESHOLD_LR,
    threshold_min: float = AVSPO_THRESHOLD_MIN,
    threshold_max: float = AVSPO_THRESHOLD_MAX,
) -> AVSPOState:
    """Update the adaptive threshold after observing a real-reward batch."""
    values = (
        state.threshold,
        global_acr,
        mean_reward,
        learning_rate,
        threshold_min,
        threshold_max,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("AVSPO state values must be finite")
    if state.previous_mean_reward is not None and not math.isfinite(
        state.previous_mean_reward
    ):
        raise ValueError("previous_mean_reward must be finite")
    if not 0.0 <= global_acr <= 1.0 or learning_rate < 0.0:
        raise ValueError("global_acr must lie in [0, 1] and learning_rate be non-negative")
    if not 0.0 <= threshold_min <= state.threshold <= threshold_max <= 1.0:
        raise ValueError("AVSPO threshold bounds are invalid")

    threshold = state.threshold
    if state.previous_mean_reward is not None:
        delta = mean_reward - state.previous_mean_reward
        direction = 1.0 if delta > 0.0 else -1.0 if delta < 0.0 else 0.0
        threshold += learning_rate * direction * (global_acr - threshold)
        threshold = min(threshold_max, max(threshold_min, threshold))
    return AVSPOState(threshold, mean_reward)


def clipped_surrogate(
    ratios: torch.Tensor,
    advantages: torch.Tensor,
    epsilon_low: float,
    epsilon_high: float,
) -> torch.Tensor:
    """Return the PPO/GRPO clipped surrogate before sign inversion."""
    if epsilon_low < 0.0 or epsilon_high < 0.0:
        raise ValueError("clip epsilons must be non-negative")
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - epsilon_low, 1.0 + epsilon_high)
    return torch.minimum(unclipped, clipped * advantages)
