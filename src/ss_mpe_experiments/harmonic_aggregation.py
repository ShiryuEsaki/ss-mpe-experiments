"""Phase K: controlled generalized harmonic aggregation."""

import importlib
import inspect
import os
import textwrap
import torch

def generalized_harmonic_amplitude(
    features_am,
    harmonic_weights,
    aggregation_order,
):
    r = float(aggregation_order)
    if r <= 0.0:
        raise ValueError("aggregation_order must be positive")
    return torch.sum(
        (features_am * harmonic_weights) ** r,
        dim=-3,
    ) ** (1.0 / r)

def install_from_environment():
    r = float(os.environ.get("HARMONIC_AGGREGATION_ORDER", "2.0"))
    if abs(r - 2.0) < 1.0e-12:
        return "native_rms_unchanged"

    module = importlib.import_module(
        "timbre_drill.framework.SS_NT"
    )
    cls = getattr(module, "SS_NT")
    source = textwrap.dedent(inspect.getsource(cls.get_all_features))

    old_sum = (
        "features_pw_h = torch.sum((features_am * harmonic_weights) "
        "** 2, dim=-3)"
    )
    old_root = (
        "features_db_h = self.sliCQ.to_decibels("
        "features_pw_h ** 0.5, rescale=False)"
    )
    new_sum = (
        "features_pw_h = torch.sum((features_am * harmonic_weights) "
        f"** {r!r}, dim=-3)"
    )
    new_root = (
        "features_db_h = self.sliCQ.to_decibels("
        f"features_pw_h ** {1.0 / r!r}, rescale=False)"
    )

    if source.count(old_sum) != 1 or source.count(old_root) != 1:
        raise RuntimeError(
            "REFUSING: SS_NT.get_all_features no longer matches "
            "the audited RMS implementation"
        )

    source = source.replace(old_sum, new_sum).replace(old_root, new_root)
    namespace = dict(module.__dict__)
    exec(compile(source, "<phase_k_generalized_aggregation>", "exec"),
         namespace)
    cls.get_all_features = namespace["get_all_features"]
    return f"generalized_r={r}"
