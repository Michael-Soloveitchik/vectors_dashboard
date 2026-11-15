"""Main orchestrator: ``project_with_centroids``.

Runs the chosen 2D reduction on ``[points; centroids; unit_balls]`` and
splits the result back, computing per-centroid base radii and pooled
covariance ellipses for the threshold-circle / σ-ellipse overlays.

See the package-level docstring (in the original projection.py) for the
derivation of the per-reduction radius calibration.
"""
from __future__ import annotations

import numpy as np

from ..gmm import CentroidSet
from ...reduction_2d import reduce_2d, ReductionCache, ReductionResult
from .combined_projection import CombinedProjection
from .nn_interpolate import _nn_interpolate


__all__ = ["project_with_centroids"]


def project_with_centroids(
    points: np.ndarray,
    centroid_set: CentroidSet,
    method: str,
    params: dict,
    cache: ReductionCache | None = None,
    n_unit_ball_samples: int = 16,
    rng_seed: int = 42,
    precomputed_2d: np.ndarray | None = None,
) -> CombinedProjection:
    """Run the chosen reduction on `[points; centroids; unit_balls]` and split back."""
    labels = centroid_set.labels()
    centroid_blocks: list[np.ndarray] = [centroid_set.centroids[l] for l in labels]
    n_per: list[int] = [b.shape[0] for b in centroid_blocks]

    S = int(n_unit_ball_samples)
    D = points.shape[1] if points.shape[0] > 0 else 0

    use_nn = (
        precomputed_2d is not None
        and centroid_blocks
    )

    if centroid_blocks:
        all_centroids = np.concatenate(centroid_blocks, axis=0)
        n_centroids_total = all_centroids.shape[0]
    else:
        all_centroids = np.empty((0, D), dtype=points.dtype)
        n_centroids_total = 0

    _pca_info_out: dict | None = None
    if use_nn:
        points_2d = precomputed_2d
        cent_2d = _nn_interpolate(points, points_2d, all_centroids)
        if S > 0 and n_centroids_total > 0:
            rng = np.random.default_rng(rng_seed)
            directions = rng.standard_normal(
                (n_centroids_total * S, D)
            ).astype(points.dtype)
            directions /= np.linalg.norm(
                directions, axis=1, keepdims=True
            ) + 1e-12
            directions *= float(np.sqrt(max(D, 1)))
            centers_repeated = np.repeat(all_centroids, S, axis=0)
            unit_ball_raw = centers_repeated + directions
            ball_2d = _nn_interpolate(points, points_2d, unit_ball_raw)
        else:
            ball_2d = np.empty((0, 2))
        res = ReductionResult(
            proj=np.concatenate([points_2d, cent_2d], axis=0),
            method=method, params_used=dict(params), cache_hit=True,
        )
        print(f"[projection] NN-path pca_info recovery gate: "
              f"method={method!r}, n_points={points.shape[0]}", flush=True)
        if (method or "").strip().lower().replace("-", "") == "pca" and points.shape[0] > 0:
            try:
                from sklearn.decomposition import PCA as _PCA
                _core = np.concatenate([points, all_centroids], axis=0) if n_centroids_total else points
                _pca_for_mesh = _PCA(n_components=2, random_state=params.get("random_state", 42))
                _pca_for_mesh.fit(_core)
                _raw_core_2d = _pca_for_mesh.transform(_core)
                _median = np.median(_raw_core_2d, axis=0, keepdims=True)
                _pca_info_out = {
                    "components": np.asarray(_pca_for_mesh.components_, dtype=np.float64),
                    "mean": np.asarray(_pca_for_mesh.mean_, dtype=np.float64),
                    "median_shift": _median.reshape(-1).astype(np.float64),
                }
                print(f"[projection] pca_info recovered: components.shape={_pca_for_mesh.components_.shape}",
                      flush=True)
            except Exception as _exc:
                print(f"[projection] pca_info recovery failed: {_exc}", flush=True)
                _pca_info_out = None
    else:
        if centroid_blocks:
            core_joined = np.concatenate([points, all_centroids], axis=0)
        else:
            core_joined = points

        extra = ("with_centroids", tuple(n_per), tuple(labels))
        res = reduce_2d(method, core_joined, params, cache=cache, cache_key_extra=extra)

        n_points = points.shape[0]
        points_2d = res.proj[:n_points]
        cent_2d = res.proj[n_points : n_points + n_centroids_total]

        ball_2d = np.empty((0, 2))
        if S > 0 and n_centroids_total > 0:
            rng = np.random.default_rng(rng_seed)
            directions = rng.standard_normal(
                (n_centroids_total * S, D)
            ).astype(points.dtype)
            directions /= np.linalg.norm(
                directions, axis=1, keepdims=True
            ) + 1e-12
            directions *= float(np.sqrt(max(D, 1)))
            centers_repeated = np.repeat(all_centroids, S, axis=0)
            unit_ball_raw = centers_repeated + directions

            method_norm = method.strip().lower().replace("-", "")
            if method_norm == "pca":
                from sklearn.decomposition import PCA as _PCA
                pca = _PCA(n_components=2, random_state=params.get("random_state", 42))
                raw_core = pca.fit_transform(core_joined)
                median = np.median(raw_core, axis=0, keepdims=True)
                ball_2d = (pca.transform(unit_ball_raw) - median).astype(np.float32)
                _pca_info_out = {
                    "components": np.asarray(pca.components_, dtype=np.float64),
                    "mean": np.asarray(pca.mean_, dtype=np.float64),
                    "median_shift": median.reshape(-1).astype(np.float64),
                }
            else:
                ball_2d = _nn_interpolate(core_joined, res.proj, unit_ball_raw)

    _member_indices: dict[str, list[int]] = {}
    _all_sids = centroid_set.member_session_ids
    overall_spread = float(np.median(
        np.linalg.norm(points_2d - np.median(points_2d, axis=0), axis=1)
    )) if points_2d.shape[0] > 2 else 1.0

    out: dict[str, np.ndarray] = {}
    out_radii: dict[str, np.ndarray] = {}
    cent_label_list: list[str] = []
    cursor = 0
    ball_cursor = 0
    for label, k in zip(labels, n_per):
        out[label] = cent_2d[cursor : cursor + k]
        cent_label_list.extend([label] * k)
        per_label_radii = np.zeros(k, dtype=np.float64)
        if S > 0:
            for j in range(k):
                c2d_j = cent_2d[cursor + j]
                ball_j = ball_2d[ball_cursor + j * S : ball_cursor + (j + 1) * S]
                if ball_j.size:
                    dists = np.linalg.norm(ball_j - c2d_j, axis=1)
                    per_label_radii[j] = float(np.median(dists))
        mean_r = float(per_label_radii[per_label_radii > 0].mean()) if np.any(per_label_radii > 0) else 0.0
        min_visible = overall_spread * 0.15
        mean_r = max(mean_r, min_visible)
        per_label_radii[:] = mean_r
        out_radii[label] = per_label_radii
        cursor += k
        ball_cursor += k * S

    out_ellipses: dict[str, list[dict]] = {}
    if S > 0:
        cursor2 = 0
        ball_cursor2 = 0
        for label, k in zip(labels, n_per):
            all_deltas = []
            centroid_centers = []
            for j in range(k):
                c2d_j = cent_2d[cursor2 + j]
                centroid_centers.append(c2d_j)
                ball_j = ball_2d[ball_cursor2 + j * S : ball_cursor2 + (j + 1) * S]
                if ball_j.size >= 2:
                    all_deltas.append(ball_j - c2d_j)
            shared_rx, shared_ry, shared_angle = 0.0, 0.0, 0.0
            if all_deltas:
                pooled = np.concatenate(all_deltas, axis=0)
                if pooled.shape[0] >= 4:
                    cov2d = np.cov(pooled.T)
                    if cov2d.shape == (2, 2):
                        eigvals, eigvecs = np.linalg.eigh(cov2d)
                        eigvals = np.maximum(eigvals, 0.0)
                        shared_angle = float(np.degrees(
                            np.arctan2(eigvecs[1, 1], eigvecs[0, 1])
                        ))
                        shared_rx = float(np.sqrt(eigvals[1]))
                        shared_ry = float(np.sqrt(eigvals[0]))
            min_ell = overall_spread * 0.08
            shared_rx = max(shared_rx, min_ell)
            shared_ry = max(shared_ry, min_ell)
            ell_list = [
                {
                    "cx": float(centroid_centers[j][0]),
                    "cy": float(centroid_centers[j][1]),
                    "rx": shared_rx,
                    "ry": shared_ry,
                    "angle": shared_angle,
                }
                for j in range(k)
            ]
            out_ellipses[label] = ell_list
            cursor2 += k
            ball_cursor2 += k * S

    all_r = np.concatenate([out_radii[l] for l in labels if l in out_radii]) if out_radii else np.array([])
    if all_r.size > 0 and np.any(all_r > 0):
        global_r = float(all_r[all_r > 0].mean())
        for label in out_radii:
            out_radii[label][:] = global_r
        print(f"[projection] global base_radius_2d = {global_r:.4f} "
              f"(overall_spread={overall_spread:.4f})", flush=True)

    if out_ellipses:
        all_valid_rx = [e["rx"] for elist in out_ellipses.values() for e in elist if e["rx"] > 0]
        all_valid_ry = [e["ry"] for elist in out_ellipses.values() for e in elist if e["ry"] > 0]
        if all_valid_rx:
            grx = sum(all_valid_rx) / len(all_valid_rx)
            gry = sum(all_valid_ry) / len(all_valid_ry)
            for elist in out_ellipses.values():
                for e in elist:
                    e["rx"] = grx
                    e["ry"] = gry

    return CombinedProjection(
        points_2d=points_2d,
        centroids_2d=out,
        centroid_labels=cent_label_list,
        result=res,
        base_radii_2d=out_radii,
        ellipses_2d=out_ellipses,
        pca_info=_pca_info_out,
        centroids_hd={l: centroid_set.centroids[l] for l in labels},
    )
