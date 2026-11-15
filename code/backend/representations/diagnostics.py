"""Post-fit diagnostic gate for ``Representation`` instances.

``self_consistency_check_fn(rep, X_probe, n_probe)`` runs the full
polarity / label-order / kernel-mass-at-diagonal audit. The
``Representation.self_consistency_check_fn`` method is a thin wrapper
that delegates here.
"""

from __future__ import annotations

import numpy as np

from .convention import closest_score_fn, pair_kernel_closest_mass_fn


def self_consistency_check_fn(
    rep,
    X_probe: np.ndarray | None = None,
    n_probe: int = 32,
) -> dict:
    """Full diagnostic run on a fitted rep. Returns a dict of booleans;
    raises ``AssertionError`` on the first failure.

    Checks:
      • ``polarity``             — ``score_xy(x, x)`` is the closest
                                    entry of each row (convention).
      • ``label_order_frozen``   — ``class_order_fn()`` unchanged
                                    across two back-to-back calls.
      • ``labels_immutable``     — ``rep._labels`` is non-writeable.
      • ``predict_polarity``     — for each predicted sample, the
                                    chosen column is ``closest_score_fn``
                                    of the raw score row.
      • ``kernel_mass_at_diag``  — ``pair_kernel_closest_mass_fn``
                                    holds on ``score_pairs(X_probe)``.
    """
    rep._require_fit()
    out: dict = {}

    # 1) Polarity invariant on score_xy (same check assert_polarity runs).
    from .math_utils import assert_polarity
    probe = (X_probe if X_probe is not None
             else np.zeros((0, 0)))             # caller must supply for full check
    if probe.size > 0:
        n = min(n_probe, probe.shape[0])
        assert_polarity(rep, probe[:n])
    out["polarity"] = True

    # 2) Label-order freeze.
    order_1 = rep.class_order_fn()
    order_2 = rep.class_order_fn()
    assert order_1 == order_2, "class_order_fn drift between calls"
    out["label_order_frozen"] = True

    # 3) Immutability of _labels.
    assert not rep._labels.flags.writeable, "_labels is writeable"
    out["labels_immutable"] = True

    # 4+5) predict polarity + kernel-mass-at-diag require probe data.
    if probe.size > 0:
        label, _, values = rep.predict(probe[:min(n_probe, probe.shape[0])])
        labels = np.asarray(rep.class_order_fn(), dtype=rep._labels.dtype)
        row_closest = closest_score_fn(values, axis=1)
        for i, lbl in enumerate(label):
            if lbl is None:
                continue
            ki = int(np.where(labels == lbl)[0][0])
            assert abs(values[i, ki] - row_closest[i]) <= 1e-9, (
                f"predict polarity drift at row {i}: "
                f"chose {lbl} ({values[i, ki]:.6f}) but row closest = "
                f"{row_closest[i]:.6f}"
            )
        out["predict_polarity"] = True

        P = rep.score_pairs(probe[:min(n_probe, probe.shape[0])])
        if not isinstance(P, tuple):
            assert pair_kernel_closest_mass_fn(np.asarray(P)), (
                "score_pairs diagonal is not the closest entry of its row"
            )
            out["kernel_mass_at_diag"] = True

    return out


__all__ = ["self_consistency_check_fn"]
