"""Low-overhead diagnostics for the SS-MPE harmonic pseudo-label losses.

The functions in this module do not change the training objective.  Gradient
statistics are computed analytically with respect to ``pitch_logits`` so that
no extra backward pass or retained autograd graph is required.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


def _weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(eps)


def _l2_norm(gradient: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(gradient.reshape(-1), ord=2)


def _cosine(
    first: torch.Tensor,
    second: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    first_flat = first.reshape(-1)
    second_flat = second.reshape(-1)
    denominator = (
        torch.linalg.vector_norm(first_flat, ord=2)
        * torch.linalg.vector_norm(second_flat, ord=2)
    )
    return torch.dot(first_flat, second_flat) / denominator.clamp_min(eps)


@torch.no_grad()
def compute_harmonic_diagnostics(
    *,
    pitch_logits: torch.Tensor,
    positive_label: torch.Tensor,
    negative_label: torch.Tensor,
    support_loss: torch.Tensor,
    harmonic_loss: torch.Tensor,
    sparsity_loss: torch.Tensor,
    multipliers: Mapping[str, float],
) -> dict[str, torch.Tensor | bool]:
    """Return scalar diagnostics without changing loss values or gradients.

    ``negative_label`` is the first-harmonic feature ``h`` used by the support
    loss.  Consequently, ``1 - h`` is its unsupported-region weight.

    Gradient norms and cosines are for each *weighted* loss with respect to
    ``pitch_logits``.  They are exact for the current definitions:

    - support: ``(1-h) * softplus(z)``
    - harmonic: ``s * softplus(-z)``
    - sparsity: ``sigmoid(z)``

    Computing logit-space gradients analytically avoids three additional
    autograd traversals through the model.
    """
    if not (
        pitch_logits.shape == positive_label.shape == negative_label.shape
    ):
        raise ValueError(
            "pitch_logits, positive_label, and negative_label must have "
            "identical shapes; got "
            f"{pitch_logits.shape}, {positive_label.shape}, "
            f"{negative_label.shape}"
        )
    if pitch_logits.ndim != 3:
        raise ValueError(
            "expected tensors shaped (batch, frequency, time), got "
            f"{pitch_logits.shape}"
        )

    # Float32 reductions are more stable than reducing autocast tensors.
    logits = pitch_logits.detach().float()
    positive = positive_label.detach().float()
    negative = negative_label.detach().float()
    probability = torch.sigmoid(logits)
    unsupported = 1.0 - negative
    eps = torch.finfo(logits.dtype).eps

    # Both support/harmonic/sparsity losses sum frequency, then average time
    # and batch.  This common factor makes the analytic logit gradients exact.
    batch_time_count = logits.shape[0] * logits.shape[-1]
    reduction_scale = 1.0 / float(batch_time_count)

    support_multiplier = float(multipliers["support_p"])
    harmonic_multiplier = float(multipliers["harmonic_p"])
    sparsity_multiplier = float(multipliers["sparsity_p"])

    support_gradient = (
        support_multiplier
        * unsupported
        * probability
        * reduction_scale
    )
    harmonic_gradient = (
        harmonic_multiplier
        * positive
        * (probability - 1.0)
        * reduction_scale
    )
    sparsity_gradient = (
        sparsity_multiplier
        * probability
        * (1.0 - probability)
        * reduction_scale
    )
    suppressive_gradient = support_gradient + sparsity_gradient
    suppressive_norm = _l2_norm(suppressive_gradient)
    harmonic_norm = _l2_norm(harmonic_gradient)

    positive_mass = positive.sum(dim=-2)

    return {
        "pseudo/positive/min": positive.min(),
        "pseudo/positive/max": positive.max(),
        "pseudo/positive/mean": positive.mean(),
        "pseudo/positive/sum_per_frame": positive_mass.mean(),
        "pseudo/positive/nonzero_ratio": (positive > 0).float().mean(),
        "pseudo/negative/mean": negative.mean(),
        # Salience-weighted fraction of the positive label that the support
        # objective simultaneously regards as unsupported.
        "pseudo/positive_negative_overlap": _weighted_mean(
            unsupported,
            positive,
            eps,
        ),
        "pseudo/positive/requires_grad": positive_label.requires_grad,
        "diagnostic/positive_probability": _weighted_mean(
            probability,
            positive,
            eps,
        ),
        "diagnostic/positive_logit": _weighted_mean(
            logits,
            positive,
            eps,
        ),
        "diagnostic/unsupported_probability": _weighted_mean(
            probability,
            unsupported,
            eps,
        ),
        "diagnostic/harmonic_loss_per_positive_mass": _weighted_mean(
            F.softplus(-logits),
            positive,
            eps,
        ),
        "weighted_loss/support": support_multiplier * support_loss.detach(),
        "weighted_loss/harmonic": harmonic_multiplier * harmonic_loss.detach(),
        "weighted_loss/sparsity": sparsity_multiplier * sparsity_loss.detach(),
        "gradient_norm/support": _l2_norm(support_gradient),
        "gradient_norm/harmonic": harmonic_norm,
        "gradient_norm/sparsity": _l2_norm(sparsity_gradient),
        "gradient_norm/support_plus_sparsity": suppressive_norm,
        "gradient_ratio/harmonic_vs_support_plus_sparsity": (
            harmonic_norm / suppressive_norm.clamp_min(eps)
        ),
        "gradient_cosine/harmonic_vs_support": _cosine(
            harmonic_gradient,
            support_gradient,
            eps,
        ),
        "gradient_cosine/harmonic_vs_sparsity": _cosine(
            harmonic_gradient,
            sparsity_gradient,
            eps,
        ),
        "gradient_cosine/harmonic_vs_support_plus_sparsity": _cosine(
            harmonic_gradient,
            suppressive_gradient,
            eps,
        ),
    }
