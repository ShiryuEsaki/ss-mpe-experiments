"""Pair-relation negative confidence and loss for SS-MPE.

All public functions accept pitch maps whose frequency dimension is selected by
``freq_dim``.  No assumption is made about a channel dimension.  The current
training tensors are shaped ``(batch, frequency, time)`` and therefore use
``freq_dim=-2``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F


PAIR_RATIOS: Mapping[str, tuple[float, float]] = {
    "2": (0.5, 2.0),
    "3": (1.0 / 3.0, 3.0),
    "5": (2.0 / 3.0, 1.5),
}


def shift_log_frequency(
    activity: torch.Tensor,
    ratio: float,
    bins_per_octave: float,
    freq_dim: int = -2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``activity`` at ``ratio * f`` using linear interpolation.

    For output frequency bin ``i``, the source position is
    ``i + bins_per_octave * log2(ratio)``.  The returned validity mask is true
    only where that continuous source position lies in ``[0, F - 1]``.  The
    time axis and every other axis are left unchanged.
    """
    if not activity.is_floating_point():
        raise TypeError("activity must be a floating-point tensor")
    if activity.ndim < 2:
        raise ValueError("activity must have at least two dimensions")
    if ratio <= 0.0 or not math.isfinite(ratio):
        raise ValueError("ratio must be finite and positive")
    if bins_per_octave <= 0.0 or not math.isfinite(bins_per_octave):
        raise ValueError("bins_per_octave must be finite and positive")

    freq_dim = freq_dim % activity.ndim
    frequency_bins = activity.shape[freq_dim]
    if frequency_bins < 1:
        raise ValueError("frequency dimension must be non-empty")

    compute_dtype = (
        torch.float64 if activity.dtype == torch.float64 else torch.float32
    )
    output_bins = torch.arange(
        frequency_bins,
        device=activity.device,
        dtype=compute_dtype,
    )
    delta = bins_per_octave * math.log2(ratio)
    source = output_bins + delta
    valid_1d = (source >= 0.0) & (source <= frequency_bins - 1)

    lower = torch.floor(source)
    upper = lower + 1.0
    upper_weight = source - lower
    lower_weight = 1.0 - upper_weight

    lower_index = lower.clamp(0, frequency_bins - 1).long()
    upper_index = upper.clamp(0, frequency_bins - 1).long()

    index_shape = [1] * activity.ndim
    index_shape[freq_dim] = frequency_bins
    expanded_shape = list(activity.shape)

    lower_index = lower_index.view(index_shape).expand(expanded_shape)
    upper_index = upper_index.view(index_shape).expand(expanded_shape)
    lower_weight = lower_weight.to(activity.dtype).view(index_shape)
    upper_weight = upper_weight.to(activity.dtype).view(index_shape)
    valid = valid_1d.view(index_shape).expand(expanded_shape)

    lower_values = torch.gather(activity, freq_dim, lower_index)
    upper_values = torch.gather(activity, freq_dim, upper_index)
    shifted = lower_weight * lower_values + upper_weight * upper_values
    shifted = torch.where(valid, shifted, torch.zeros_like(shifted))
    return shifted, valid


def pair_positive(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Soft AND for a symmetric relation pair: ``min(first, second)``."""
    return torch.minimum(first, second)


def pair_negative(
    first: torch.Tensor,
    second: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Soft one-sided mismatch confidence for a symmetric relation pair."""
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    return (
        torch.maximum(first, second)
        * torch.abs(first - second)
        / (first + second + eps)
    )


def _top2_mean_valid(
    values: torch.Tensor,
    valid: torch.Tensor,
    min_valid_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if values.shape != valid.shape or values.shape[0] != 3:
        raise ValueError("values and valid must be shaped (3, ...)")
    if min_valid_pairs != 2:
        raise ValueError("the first implementation requires min_valid_pairs=2")

    valid_count = valid.sum(dim=0)
    valid_position = valid_count >= min_valid_pairs
    masked = values.masked_fill(~valid, float("-inf"))
    top2 = torch.topk(masked, k=2, dim=0).values.mean(dim=0)
    top2 = torch.where(valid_position, top2, torch.zeros_like(top2))
    return top2, valid_count, valid_position


def _mean_valid(
    values: torch.Tensor,
    valid: torch.Tensor,
    valid_count: torch.Tensor,
    valid_position: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    result = (values * valid.to(values.dtype)).sum(dim=0)
    result = result / valid_count.to(values.dtype).clamp_min(eps)
    return torch.where(valid_position, result, torch.zeros_like(result))


@torch.no_grad()
def build_pair_relation_maps(
    activity: torch.Tensor,
    *,
    bins_per_octave: float,
    freq_dim: int = -2,
    min_valid_pairs: int = 2,
    eps: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    """Build six relation maps, three pairs, and their aggregate scores."""
    activity = activity.detach()
    relation: dict[str, torch.Tensor] = {}
    relation_valid: dict[str, torch.Tensor] = {}

    ratio_names = {
        "half": 0.5,
        "double": 2.0,
        "third": 1.0 / 3.0,
        "triple": 3.0,
        "two_thirds": 2.0 / 3.0,
        "three_halves": 1.5,
    }
    for name, ratio in ratio_names.items():
        relation[name], relation_valid[name] = shift_log_frequency(
            activity,
            ratio,
            bins_per_octave,
            freq_dim,
        )

    pair_inputs = {
        "2": ("half", "double"),
        "3": ("third", "triple"),
        "5": ("two_thirds", "three_halves"),
    }
    positives = []
    negatives = []
    pair_valids = []

    result = {f"activity_{name}": value for name, value in relation.items()}
    result.update(
        {f"valid_{name}": value for name, value in relation_valid.items()}
    )

    for pair_name, (first_name, second_name) in pair_inputs.items():
        first = relation[first_name]
        second = relation[second_name]
        valid = relation_valid[first_name] & relation_valid[second_name]
        positive = pair_positive(first, second)
        negative = pair_negative(first, second, eps)
        positive = torch.where(valid, positive, torch.zeros_like(positive))
        negative = torch.where(valid, negative, torch.zeros_like(negative))
        result[f"p{pair_name}"] = positive
        result[f"n{pair_name}"] = negative
        result[f"v{pair_name}"] = valid
        positives.append(positive)
        negatives.append(negative)
        pair_valids.append(valid)

    positive_stack = torch.stack(positives, dim=0)
    negative_stack = torch.stack(negatives, dim=0)
    valid_stack = torch.stack(pair_valids, dim=0)
    r_pos, valid_count, m_valid = _top2_mean_valid(
        positive_stack,
        valid_stack,
        min_valid_pairs,
    )
    r_neg = _mean_valid(
        negative_stack,
        valid_stack,
        valid_count,
        m_valid,
        eps,
    )
    result.update(
        {
            "r_pos": r_pos,
            "r_neg": r_neg,
            "valid_count": valid_count,
            "m_valid": m_valid,
        }
    )
    return result


@torch.no_grad()
def build_pair_negative_confidence(
    positive_label: torch.Tensor,
    *,
    bins_per_octave: float,
    freq_dim: int = -2,
    candidate_threshold: float = 0.1,
    min_valid_pairs: int = 2,
    eps: float = 1.0e-8,
    validate_range: bool = False,
) -> dict[str, torch.Tensor]:
    """Return detached pair maps, masks, and absolute negative confidence."""
    if not 0.0 <= candidate_threshold <= 1.0:
        raise ValueError("candidate_threshold must be in [0, 1]")
    if validate_range:
        observed_min = float(positive_label.detach().min())
        observed_max = float(positive_label.detach().max())
        if observed_min < 0.0 or observed_max > 1.0:
            raise ValueError(
                "positive activity must lie in [0, 1]; observed "
                f"[{observed_min}, {observed_max}]"
            )

    activity = positive_label.detach().clamp(0.0, 1.0)
    result = build_pair_relation_maps(
        activity,
        bins_per_octave=bins_per_octave,
        freq_dim=freq_dim,
        min_valid_pairs=min_valid_pairs,
        eps=eps,
    )
    candidate = positive_label.detach() >= candidate_threshold
    base_mask = candidate & result["m_valid"]
    w_minus = (
        base_mask.to(activity.dtype)
        * result["r_neg"]
        * (1.0 - result["r_pos"])
    ).clamp(0.0, 1.0).detach()
    result.update(
        {
            "activity": activity,
            "m_candidate": candidate,
            "base_mask": base_mask,
            "w_minus": w_minus,
        }
    )
    return result


def pair_negative_loss(
    pitch_logits: torch.Tensor,
    w_minus: torch.Tensor,
    base_mask: torch.Tensor,
) -> torch.Tensor:
    """Absolute-confidence weighted negative BCE normalized by candidates."""
    if pitch_logits.shape != w_minus.shape or pitch_logits.shape != base_mask.shape:
        raise ValueError("pitch_logits, w_minus, and base_mask must match")
    # Compute softplus and reductions in float32 under CUDA
    # autocast. The cast remains differentiable, so gradients still
    # reach the original logits.
    stable_logits = pitch_logits.float()
    weight = w_minus.detach().float()
    denominator = base_mask.float().sum().clamp_min(1.0)
    return (weight * F.softplus(stable_logits)).sum() / denominator


def combine_pair_negative_loss(
    existing_loss: torch.Tensor,
    pair_loss: torch.Tensor,
    *,
    alpha_pair_neg: float,
    enabled: bool,
) -> torch.Tensor:
    """Return the original tensor unchanged when the feature is disabled."""
    if not enabled:
        return existing_loss
    return existing_loss + float(alpha_pair_neg) * pair_loss

@torch.no_grad()
def summarize_pair_negative(
    maps: Mapping[str, torch.Tensor],
    loss: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Summarize pair maps without retaining a training graph."""
    reference = maps["w_minus"].detach().float()
    zero = reference.new_zeros(())

    def masked_values(name: str, mask_name: str) -> torch.Tensor:
        values = maps[name].detach().float()
        mask = maps[mask_name].detach().bool()
        return values[mask]

    def mean_or_zero(values: torch.Tensor) -> torch.Tensor:
        return values.mean() if values.numel() else zero

    def max_or_zero(values: torch.Tensor) -> torch.Tensor:
        return values.max() if values.numel() else zero

    def quantile_or_zero(
        values: torch.Tensor,
        q: float,
    ) -> torch.Tensor:
        return torch.quantile(values, q) if values.numel() else zero

    base = maps["base_mask"].detach().bool()
    w_values = reference[base]
    valid_position = maps["m_valid"].detach().bool()
    r_pos_values = (
        maps["r_pos"].detach().float()[valid_position]
    )
    r_neg_values = (
        maps["r_neg"].detach().float()[valid_position]
    )
    valid_count = maps["valid_count"].detach()

    metrics = {
        "pair_neg/loss": loss.detach().float(),
        "pair_neg/w_mean": mean_or_zero(w_values),
        "pair_neg/w_max": max_or_zero(w_values),
        "pair_neg/w_p50": quantile_or_zero(w_values, 0.50),
        "pair_neg/w_p90": quantile_or_zero(w_values, 0.90),
        "pair_neg/w_nonzero_ratio": (
            (w_values > 0).float().mean()
            if w_values.numel()
            else zero
        ),
        "pair_neg/r_pos_mean": mean_or_zero(r_pos_values),
        "pair_neg/r_pos_p90": quantile_or_zero(
            r_pos_values, 0.90
        ),
        "pair_neg/r_neg_mean": mean_or_zero(r_neg_values),
        "pair_neg/r_neg_p90": quantile_or_zero(
            r_neg_values, 0.90
        ),
        "pair_neg/p2_mean": mean_or_zero(
            masked_values("p2", "v2")
        ),
        "pair_neg/p3_mean": mean_or_zero(
            masked_values("p3", "v3")
        ),
        "pair_neg/p5_mean": mean_or_zero(
            masked_values("p5", "v5")
        ),
        "pair_neg/n2_mean": mean_or_zero(
            masked_values("n2", "v2")
        ),
        "pair_neg/n3_mean": mean_or_zero(
            masked_values("n3", "v3")
        ),
        "pair_neg/n5_mean": mean_or_zero(
            masked_values("n5", "v5")
        ),
        "pair_neg/candidate_ratio": (
            maps["m_candidate"].detach().float().mean()
        ),
        "pair_neg/valid_ratio": valid_position.float().mean(),
    }

    for count in range(4):
        metrics[
            f"pair_neg/valid_count_{count}_ratio"
        ] = (valid_count == count).float().mean()

    return metrics
