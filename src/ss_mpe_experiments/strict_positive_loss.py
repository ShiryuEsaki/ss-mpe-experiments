# STRICT_UPPER_POSITIVE_V1
# STRICT_UPPER_POSITIVE_GATE_DIAGNOSTICS_V2
# STRICT_UPPER_POSITIVE_RELATIVE_BASE_GT_V3

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


POSITIVE_MODES = {
    "peak_only",
    "relative_energy_only",
    "relative_energy_base",
    "peak_relative_energy",
    "strict_upper_relative_energy",
}


def _validate_bft(name: str, value: torch.Tensor) -> None:
    if value.ndim != 3:
        raise ValueError(
            f"{name} must have shape (B, F, T), got {tuple(value.shape)}"
        )


def _harmonic_index(
    harmonics: Sequence[float],
    target: float,
    tolerance: float = 1.0e-6,
) -> int:
    matches = [
        index
        for index, value in enumerate(harmonics)
        if abs(float(value) - target) <= tolerance
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one h={target} channel, "
            f"got harmonics={list(harmonics)}"
        )
    return matches[0]


def _band_energy(
    amplitude: torch.Tensor,
    radius_bins: int,
    reduction: str,
) -> torch.Tensor:
    """Convert amplitude to power and aggregate along frequency."""
    _validate_bft("amplitude", amplitude)

    if radius_bins < 0:
        raise ValueError("radius_bins must be non-negative")
    if reduction not in {"sum", "max"}:
        raise ValueError("reduction must be 'sum' or 'max'")

    power = amplitude.detach().float().square()

    if radius_bins == 0:
        return power

    kernel = 2 * radius_bins + 1
    power_4d = power.unsqueeze(1)

    if reduction == "sum":
        energy = F.avg_pool2d(
            power_4d,
            kernel_size=(kernel, 1),
            stride=1,
            padding=(radius_bins, 0),
        ) * kernel
    else:
        energy = F.max_pool2d(
            power_4d,
            kernel_size=(kernel, 1),
            stride=1,
            padding=(radius_bins, 0),
        )

    return energy.squeeze(1)


def _sample_log_frequency(
    values: torch.Tensor,
    *,
    ratio: float,
    bins_per_octave: float,
    edge_radius: int,
    nearest: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample values at ratio*f for every output base-frequency bin.

    Out-of-range positions are zeroed and accompanied by a false valid mask.
    """
    _validate_bft("values", values)

    if ratio <= 0.0 or not math.isfinite(ratio):
        raise ValueError("ratio must be finite and positive")
    if bins_per_octave <= 0.0 or not math.isfinite(bins_per_octave):
        raise ValueError("bins_per_octave must be finite and positive")
    if edge_radius < 0:
        raise ValueError("edge_radius must be non-negative")

    batch, frequency_bins, frames = values.shape
    positions = torch.arange(
        frequency_bins,
        device=values.device,
        dtype=torch.float32,
    )
    source = positions + bins_per_octave * math.log2(ratio)

    valid_1d = (
        (source >= float(edge_radius))
        & (
            source
            <= float(frequency_bins - 1 - edge_radius)
        )
    )

    if nearest:
        index = source.round().clamp(
            0,
            frequency_bins - 1,
        ).long()
        gather_index = index.view(1, -1, 1).expand(
            batch,
            frequency_bins,
            frames,
        )
        sampled = torch.gather(values, 1, gather_index)
    else:
        lower = torch.floor(source)
        upper = torch.ceil(source)

        lower_index = lower.clamp(
            0,
            frequency_bins - 1,
        ).long()
        upper_index = upper.clamp(
            0,
            frequency_bins - 1,
        ).long()

        lower_gather = lower_index.view(1, -1, 1).expand(
            batch,
            frequency_bins,
            frames,
        )
        upper_gather = upper_index.view(1, -1, 1).expand(
            batch,
            frequency_bins,
            frames,
        )

        lower_value = torch.gather(values, 1, lower_gather)
        upper_value = torch.gather(values, 1, upper_gather)

        upper_weight = (source - lower).view(1, -1, 1)
        lower_weight = 1.0 - upper_weight

        sampled = (
            lower_value * lower_weight
            + upper_value * upper_weight
        )

    valid = valid_1d.view(1, -1, 1).expand(
        batch,
        frequency_bins,
        frames,
    )

    sampled = torch.where(
        valid,
        sampled,
        torch.zeros_like(sampled),
    )
    return sampled, valid


def _relative_db(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return 10.0 * torch.log10(
        (numerator.float() + eps)
        / (denominator.float() + eps)
    )


def _local_peak_map(
    energy: torch.Tensor,
    *,
    radius_bins: int,
    prominence_threshold_db: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local-peak mask and dB prominence."""
    _validate_bft("energy", energy)

    if radius_bins < 1:
        raise ValueError("peak radius must be at least 1")

    kernel = 2 * radius_bins + 1
    padded = F.pad(
        energy.float().unsqueeze(1),
        (0, 0, radius_bins, radius_bins),
        mode="replicate",
    ).squeeze(1)

    windows = padded.unfold(1, kernel, 1)
    center = windows[..., radius_bins]
    left_max = windows[..., :radius_bins].amax(dim=-1)
    right_max = windows[..., radius_bins + 1:].amax(dim=-1)
    neighbor_max = torch.maximum(left_max, right_max)

    prominence_db = _relative_db(
        center,
        neighbor_max,
        eps,
    )
    peak = (
        (center >= left_max)
        & (center >= right_max)
        & (prominence_db >= prominence_threshold_db)
    )
    return peak, prominence_db


def _local_noise_floor(
    energy: torch.Tensor,
    radius_bins: int,
) -> torch.Tensor:
    """Median local energy excluding the center bin."""
    _validate_bft("energy", energy)

    if radius_bins < 1:
        raise ValueError("noise radius must be at least 1")

    kernel = 2 * radius_bins + 1
    padded = F.pad(
        energy.float().unsqueeze(1),
        (0, 0, radius_bins, radius_bins),
        mode="replicate",
    ).squeeze(1)

    windows = padded.unfold(1, kernel, 1)
    neighbors = torch.cat(
        (
            windows[..., :radius_bins],
            windows[..., radius_bins + 1:],
        ),
        dim=-1,
    )
    return neighbors.median(dim=-1).values


@torch.no_grad()
def build_strict_upper_positive(
    hcqt_amplitude: torch.Tensor,
    positive_label: torch.Tensor,
    *,
    harmonics: Sequence[float],
    bins_per_octave: float,
    candidate_threshold: float,
    threshold_base_snr_db: float,
    threshold_db_2x: float,
    threshold_db_3x: float,
    threshold_db_1p5x: float,
    harmonic_band_radius_bins: int = 0,
    harmonic_tolerance_cents: float | None = None,
    harmonic_energy_reduction: str = "sum",
    peak_radius_bins: int = 2,
    peak_prominence_threshold_db: float = 3.0,
    relative_energy_eps: float = 1.0e-8,
    positive_mode: str = "strict_upper_relative_energy",
) -> dict[str, torch.Tensor]:
    """
    Build a narrow high-confidence positive mask from upper harmonics only.

    hcqt_amplitude must be the linear relative-amplitude tensor produced
    inside SS_NT.get_all_features(), shaped (B, H, F, T).

    A failed strict condition remains unknown/ignore. It is not converted
    into a negative target.
    """
    if hcqt_amplitude.ndim != 4:
        raise ValueError(
            "hcqt_amplitude must have shape (B, H, F, T)"
        )
    _validate_bft("positive_label", positive_label)

    if (
        hcqt_amplitude.shape[0] != positive_label.shape[0]
        or hcqt_amplitude.shape[2:] != positive_label.shape[1:]
    ):
        raise ValueError(
            "HCQT and positive-label batch/frequency/time shapes must match"
        )
    if not 0.0 <= candidate_threshold <= 1.0:
        raise ValueError("candidate_threshold must be in [0, 1]")
    if relative_energy_eps <= 0.0:
        raise ValueError("relative_energy_eps must be positive")
    if positive_mode not in POSITIVE_MODES:
        raise ValueError(
            f"Unknown positive_mode={positive_mode!r}; "
            f"expected one of {sorted(POSITIVE_MODES)}"
        )

    effective_band_radius = int(harmonic_band_radius_bins)
    if harmonic_tolerance_cents is not None:
        if harmonic_tolerance_cents < 0.0:
            raise ValueError(
                "harmonic_tolerance_cents must be non-negative"
            )
        cents_per_bin = 1200.0 / float(bins_per_octave)
        effective_band_radius = int(
            math.ceil(harmonic_tolerance_cents / cents_per_bin)
        )

    h1_index = _harmonic_index(harmonics, 1.0)
    base_amplitude = (
        hcqt_amplitude[:, h1_index]
        .detach()
        .float()
        .clamp_min(0.0)
    )

    base_energy = _band_energy(
        base_amplitude,
        effective_band_radius,
        harmonic_energy_reduction,
    )

    edge_radius = max(
        effective_band_radius,
        int(peak_radius_bins),
    )

    energy_1x = base_energy
    energy_1p5x, valid_1p5x = _sample_log_frequency(
        base_energy,
        ratio=1.5,
        bins_per_octave=bins_per_octave,
        edge_radius=edge_radius,
        nearest=False,
    )
    energy_2x, valid_2x = _sample_log_frequency(
        base_energy,
        ratio=2.0,
        bins_per_octave=bins_per_octave,
        edge_radius=edge_radius,
        nearest=False,
    )
    energy_3x, valid_3x = _sample_log_frequency(
        base_energy,
        ratio=3.0,
        bins_per_octave=bins_per_octave,
        edge_radius=edge_radius,
        nearest=False,
    )

    batch, frequency_bins, frames = positive_label.shape
    frequency_index = torch.arange(
        frequency_bins,
        device=positive_label.device,
    )
    valid_1d = (
        (frequency_index >= edge_radius)
        & (frequency_index < frequency_bins - edge_radius)
    )
    valid_1x = valid_1d.view(1, -1, 1).expand(
        batch,
        frequency_bins,
        frames,
    )

    peak_map, peak_prominence_db = _local_peak_map(
        base_energy,
        radius_bins=peak_radius_bins,
        prominence_threshold_db=peak_prominence_threshold_db,
        eps=relative_energy_eps,
    )

    peak_mask_1p5x, peak_valid_1p5x = _sample_log_frequency(
        peak_map,
        ratio=1.5,
        bins_per_octave=bins_per_octave,
        edge_radius=edge_radius,
        nearest=True,
    )
    peak_mask_2x, peak_valid_2x = _sample_log_frequency(
        peak_map,
        ratio=2.0,
        bins_per_octave=bins_per_octave,
        edge_radius=edge_radius,
        nearest=True,
    )
    peak_mask_3x, peak_valid_3x = _sample_log_frequency(
        peak_map,
        ratio=3.0,
        bins_per_octave=bins_per_octave,
        edge_radius=edge_radius,
        nearest=True,
    )

    valid_1p5x = valid_1p5x & peak_valid_1p5x
    valid_2x = valid_2x & peak_valid_2x
    valid_3x = valid_3x & peak_valid_3x

    db_1p5x = _relative_db(
        energy_1p5x,
        energy_1x,
        relative_energy_eps,
    )
    db_2x = _relative_db(
        energy_2x,
        energy_1x,
        relative_energy_eps,
    )
    db_3x = _relative_db(
        energy_3x,
        energy_1x,
        relative_energy_eps,
    )

    noise_1x = _local_noise_floor(
        energy_1x,
        radius_bins=peak_radius_bins,
    )
    base_snr_db = _relative_db(
        energy_1x,
        noise_1x,
        relative_energy_eps,
    )

    candidate_mask = (
        positive_label.detach().float() >= candidate_threshold
    )
    valid_all = (
        valid_1x
        & valid_1p5x
        & valid_2x
        & valid_3x
    )
    base_valid = base_snr_db >= threshold_base_snr_db

    peak_gate = (
        peak_mask_2x
        & peak_mask_3x
        & ~peak_mask_1p5x
    )

    relative_mask_2x = db_2x >= threshold_db_2x
    relative_mask_3x = db_3x >= threshold_db_3x
    relative_mask_1p5x = db_1p5x < threshold_db_1p5x

    relative_gate = (
        relative_mask_2x
        & relative_mask_3x
        & relative_mask_1p5x
    )

    common = candidate_mask & valid_all

    # These four masks expose the A/B/C/D ablations without changing
    # which mode is used for the actual loss.
    positive_peak_only = common & peak_gate
    positive_relative_only = common & relative_gate
    positive_relative_base = (
        common
        & base_valid
        & relative_gate
    )
    positive_peak_relative = common & peak_gate & relative_gate
    positive_strict_upper = (
        common
        & base_valid
        & peak_gate
        & relative_gate
    )

    if positive_mode == "peak_only":
        strict_positive = positive_peak_only
    elif positive_mode == "relative_energy_only":
        strict_positive = positive_relative_only
    elif positive_mode == "relative_energy_base":
        strict_positive = positive_relative_base
    elif positive_mode == "peak_relative_energy":
        strict_positive = positive_peak_relative
    else:
        strict_positive = positive_strict_upper

    # Sequential D-mode gates for diagnostics.
    gate_candidate = candidate_mask
    gate_valid = gate_candidate & valid_all
    gate_base = gate_valid & base_valid
    gate_peak_2x = gate_base & peak_mask_2x
    gate_peak_3x = gate_peak_2x & peak_mask_3x
    gate_no_peak_1p5x = gate_peak_3x & ~peak_mask_1p5x
    gate_relative_2x = (
        gate_no_peak_1p5x
        & (db_2x >= threshold_db_2x)
    )
    gate_relative_3x = (
        gate_relative_2x
        & (db_3x >= threshold_db_3x)
    )
    gate_relative_1p5x = (
        gate_relative_3x
        & (db_1p5x < threshold_db_1p5x)
    )

    strict_positive = strict_positive.bool().detach()

    return {
        "candidate_mask": candidate_mask,
        "valid_1x": valid_1x,
        "valid_1p5x": valid_1p5x,
        "valid_2x": valid_2x,
        "valid_3x": valid_3x,
        "base_valid": base_valid,
        "energy_1x": energy_1x,
        "energy_1p5x": energy_1p5x,
        "energy_2x": energy_2x,
        "energy_3x": energy_3x,
        "db_1p5x": db_1p5x,
        "db_2x": db_2x,
        "db_3x": db_3x,
        "noise_1x": noise_1x,
        "base_snr_db": base_snr_db,
        "peak_mask_1p5x": peak_mask_1p5x,
        "peak_mask_2x": peak_mask_2x,
        "peak_mask_3x": peak_mask_3x,
        "peak_prominence_db": peak_prominence_db,
        "valid_all": valid_all,
        "peak_gate": peak_gate,
        "relative_mask_2x": relative_mask_2x,
        "relative_mask_3x": relative_mask_3x,
        "relative_mask_1p5x": relative_mask_1p5x,
        "relative_gate": relative_gate,
        "positive_peak_only": positive_peak_only,
        "positive_relative_only": positive_relative_only,
        "positive_relative_base": positive_relative_base,
        "positive_peak_relative": positive_peak_relative,
        "positive_strict_upper": positive_strict_upper,
        "strict_positive": strict_positive,
        "ignore_mask": candidate_mask & ~strict_positive,
        "gate_candidate": gate_candidate,
        "gate_valid": gate_valid,
        "gate_base": gate_base,
        "gate_peak_2x": gate_peak_2x,
        "gate_peak_3x": gate_peak_3x,
        "gate_no_peak_1p5x": gate_no_peak_1p5x,
        "gate_relative_2x": gate_relative_2x,
        "gate_relative_3x": gate_relative_3x,
        "gate_relative_1p5x": gate_relative_1p5x,
        "effective_band_radius_bins": torch.tensor(
            effective_band_radius,
            device=positive_label.device,
        ),
    }


def strict_positive_loss(
    pitch_logits: torch.Tensor,
    strict_positive: torch.Tensor,
) -> torch.Tensor:
    if pitch_logits.shape != strict_positive.shape:
        raise ValueError(
            "pitch_logits and strict_positive must have equal shapes"
        )

    stable_logits = pitch_logits.float()
    mask = strict_positive.detach().bool()
    denominator = mask.float().sum().clamp_min(1.0)

    return (
        F.softplus(-stable_logits)
        * mask.to(stable_logits.dtype)
    ).sum() / denominator


@torch.no_grad()
def summarize_strict_upper_positive(
    maps: Mapping[str, torch.Tensor],
    loss: torch.Tensor,
) -> dict[str, torch.Tensor]:
    reference = maps["strict_positive"].float()
    total = reference.new_tensor(float(reference.numel()))
    batch = reference.shape[0]

    metrics: dict[str, torch.Tensor] = {
        "strict_positive/loss": loss.detach().float(),
    }

    count_names = (
        "candidate_mask",
        "valid_1p5x",
        "valid_2x",
        "valid_3x",
        "base_valid",
        "peak_mask_2x",
        "peak_mask_3x",
        "peak_gate",
        "relative_mask_2x",
        "relative_mask_3x",
        "relative_mask_1p5x",
        "relative_gate",
        "positive_peak_only",
        "positive_relative_only",
        "positive_relative_base",
        "positive_peak_relative",
        "positive_strict_upper",
        "strict_positive",
        "gate_candidate",
        "gate_valid",
        "gate_base",
        "gate_peak_2x",
        "gate_peak_3x",
        "gate_no_peak_1p5x",
        "gate_relative_2x",
        "gate_relative_3x",
        "gate_relative_1p5x",
    )

    for name in count_names:
        count = maps[name].detach().float().sum()
        metrics[f"strict_positive/{name}_count"] = count
        metrics[f"strict_positive/{name}_ratio"] = count / total

    # Distribution measurements use at most 200k deterministic
    # samples per batch to keep the diagnostic inexpensive.
    distribution_mask = (
        maps["candidate_mask"].detach().bool()
        & maps["valid_all"].detach().bool()
    )

    for value_name in (
        "base_snr_db",
        "db_2x",
        "db_3x",
        "db_1p5x",
    ):
        values = maps[value_name].detach().float()[distribution_mask]

        if values.numel() == 0:
            values = reference.new_zeros(1)
        elif values.numel() > 200000:
            stride = (
                values.numel() + 200000 - 1
            ) // 200000
            values = values[::stride]

        for quantile_name, quantile in (
            ("p01", 0.01),
            ("p10", 0.10),
            ("p25", 0.25),
            ("p50", 0.50),
            ("p75", 0.75),
            ("p90", 0.90),
            ("p99", 0.99),
        ):
            metrics[
                f"strict_positive/distribution_{value_name}_{quantile_name}"
            ] = torch.quantile(values, quantile)

    positive_per_batch = (
        maps["strict_positive"]
        .reshape(batch, -1)
        .sum(dim=1)
    )
    metrics["strict_positive/zero_positive_batch_rate"] = (
        positive_per_batch.eq(0).float().mean()
    )
    return metrics


def _dilate_frequency_mask(
    mask: torch.Tensor,
    radius_bins: int,
) -> torch.Tensor:
    """Dilate a BFT boolean mask along frequency only."""
    _validate_bft("mask", mask)

    if radius_bins < 0:
        raise ValueError("radius_bins must be non-negative")

    mask = mask.detach().bool()

    if radius_bins == 0:
        return mask

    return (
        F.max_pool2d(
            mask.float().unsqueeze(1),
            kernel_size=(2 * radius_bins + 1, 1),
            stride=1,
            padding=(radius_bins, 0),
        )
        .squeeze(1)
        .bool()
    )


@torch.no_grad()
def summarize_strict_positive_ground_truth(
    strict_positive: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    bins_per_octave: float,
) -> dict[str, torch.Tensor]:
    """
    Evaluate anchors against GT without using GT to construct the anchors.

    Frequency tolerances:
        0 bins = exact
        1 bin  = 20 cents at 60 bins/octave
        2 bins = 40 cents at 60 bins/octave
    """
    _validate_bft("strict_positive", strict_positive)
    _validate_bft("ground_truth", ground_truth)

    if strict_positive.shape != ground_truth.shape:
        raise ValueError(
            "Strict-positive and GT shapes must match exactly; "
            f"strict={tuple(strict_positive.shape)}, "
            f"gt={tuple(ground_truth.shape)}"
        )

    anchor = strict_positive.detach().bool()
    gt = ground_truth.detach().float().gt(0.5)

    zero = ground_truth.new_zeros((), dtype=torch.float32)

    def count(mask: torch.Tensor) -> torch.Tensor:
        return mask.detach().float().sum()

    def ratio(
        numerator: torch.Tensor,
        denominator: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            denominator > 0,
            numerator / denominator,
            zero,
        )

    anchor_count = count(anchor)
    gt_count = count(gt)

    gt_tol1 = _dilate_frequency_mask(gt, 1)
    gt_tol2 = _dilate_frequency_mask(gt, 2)

    anchor_tol2 = _dilate_frequency_mask(anchor, 2)

    match_exact = count(anchor & gt)
    match_tol1 = count(anchor & gt_tol1)
    match_tol2 = count(anchor & gt_tol2)

    gt_covered_tol2 = count(gt & anchor_tol2)

    false_anchor_tol2 = anchor & ~gt_tol2
    false_anchor_count = count(false_anchor_tol2)

    # For an anchor at f:
    # GT at f/2 means an octave-up anchor error.
    # GT at 2f means an octave-down anchor error.
    gt_half, valid_half = _sample_log_frequency(
        gt_tol2,
        ratio=0.5,
        bins_per_octave=bins_per_octave,
        edge_radius=0,
        nearest=True,
    )
    gt_double, valid_double = _sample_log_frequency(
        gt_tol2,
        ratio=2.0,
        bins_per_octave=bins_per_octave,
        edge_radius=0,
        nearest=True,
    )

    octave_up = count(
        false_anchor_tol2
        & valid_half
        & gt_half.bool()
    )
    octave_down = count(
        false_anchor_tol2
        & valid_double
        & gt_double.bool()
    )

    return {
        "strict_positive_gt/anchor_count": anchor_count,
        "strict_positive_gt/gt_count": gt_count,
        "strict_positive_gt/match_exact_count": match_exact,
        "strict_positive_gt/match_tol1_count": match_tol1,
        "strict_positive_gt/match_tol2_count": match_tol2,
        "strict_positive_gt/gt_covered_tol2_count": (
            gt_covered_tol2
        ),
        "strict_positive_gt/false_anchor_tol2_count": (
            false_anchor_count
        ),
        "strict_positive_gt/octave_up_count": octave_up,
        "strict_positive_gt/octave_down_count": octave_down,
        "strict_positive_gt/precision_exact": ratio(
            match_exact,
            anchor_count,
        ),
        "strict_positive_gt/precision_tol1": ratio(
            match_tol1,
            anchor_count,
        ),
        "strict_positive_gt/precision_tol2": ratio(
            match_tol2,
            anchor_count,
        ),
        "strict_positive_gt/gt_coverage_tol2": ratio(
            gt_covered_tol2,
            gt_count,
        ),
        "strict_positive_gt/octave_up_false_ratio": ratio(
            octave_up,
            false_anchor_count,
        ),
        "strict_positive_gt/octave_down_false_ratio": ratio(
            octave_down,
            false_anchor_count,
        ),
    }
