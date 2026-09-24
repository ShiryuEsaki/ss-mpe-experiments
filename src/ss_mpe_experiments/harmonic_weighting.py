"""Phase K: label-free harmonic pseudo-label weights."""

import math
import os
import torch

BASELINE_POWER_SUM = 0.5043289065361023

def _selected_orders(harmonics, max_order):
    text = os.environ.get("HARMONIC_INCLUDED_ORDERS", "").strip()
    if text:
        requested = tuple(
            sorted({int(token.strip()) for token in text.split(",") if token.strip()})
        )
        if not requested or min(requested) < 1:
            raise ValueError("HARMONIC_INCLUDED_ORDERS must contain positive integers")
    else:
        requested = tuple(range(1, int(max_order) + 1))

    available = {
        int(round(float(h)))
        for h in harmonics
        if float(h) >= 1.0
        and math.isclose(float(h), round(float(h)), abs_tol=1.0e-8)
    }
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(f"requested harmonics unavailable: {missing}")
    return requested

def make_harmonic_amplitude_weights(
    harmonics,
    *,
    max_order,
    power_exponent,
    normalization,
    device,
):
    if normalization != "baseline_power_sum":
        raise ValueError(
            "Phase K requires normalization='baseline_power_sum'"
        )

    q = float(power_exponent)
    if int(max_order) < 1 or q < 0.0:
        raise ValueError("max_order must be >=1 and q must be non-negative")

    values = [float(h) for h in harmonics]
    orders = _selected_orders(values, max_order)
    selected = {
        order: next(
            index for index, value in enumerate(values)
            if math.isclose(value, float(order), abs_tol=1.0e-8)
        )
        for order in orders
    }

    raw_power = {order: order ** (-q) for order in orders}
    scale = math.sqrt(
        BASELINE_POWER_SUM / sum(raw_power.values())
    )

    result = [0.0] * len(values)
    for order, index in selected.items():
        result[index] = scale * math.sqrt(raw_power[order])

    return torch.tensor(
        result, dtype=torch.float32, device=device
    ).view(-1, 1, 1)

def get_effective_power_weights(harmonic_weights):
    return (
        harmonic_weights.squeeze(-1)
        .squeeze(-1)
        .detach()
        .pow(2)
    )
