"""Geometric opening-rim extraction + PCA debug visualization for deformable bracelets.

No fixed vertex indices: rim candidates are chosen from current nodal positions each call.
Designed for reset / spawn frame (before large deformation).

Usage (inside a running ReachDeformableBraceletEnv)::

    from bracelet_opening_pca_viz import compute_bracelet_opening_pca

    result = compute_bracelet_opening_pca(env, env_id=0, draw=True)
    print(result.summary())

Limitations (see :func:`extract_opening_rim_points_geometric`):
- Assumes a cuff / torus-like mesh with one dominant long axis and one hand-facing opening.
- Heavy self-intersection, torn topology, or wrong ``opening_toward_env_point`` can mis-label the opening end.
- Hex resolution changes vertex count but not this pipeline (no index cache).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)

WORLD_AXIS_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
WORLD_AXIS_VECTORS = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)


@dataclass
class OpeningRimExtractConfig:
    """Geometric rim extraction (no fixed nodal indices).

    Pipeline
    --------
    1. Cuff long axis ``a``; opening-end cap → opening plane ``(p0, n)``.
    2. **Plane distance**: keep vertices in an expanded end cap within ``plane_band_max``.
    3. Project to plane; **radial outer** annulus (soft inner cut only; no global ``r_max`` fraction).
    4. **Angular bins**: per azimuth, pick outermost vertex → rim ring sample.

    Do **not** use a high ``ring_radius_min_frac * r_max`` cut on ellipses: N/S have smaller
    in-plane radius than E/W and would be dropped (left/right strips only).
    """

    end_slice_ratio: float = 0.12
    #: Multiply ``end_slice_ratio`` for the plane-near cuff band along ``a``.
    end_cap_expand: float = 2.5
    #: Plane-distance quantile on expanded end cap (|dot(p-p0,n)|).
    plane_band_quantile: float = 0.97
    plane_band_scale: float = 2.5
    #: Number of azimuth bins in the opening plane for outer-rim picking.
    angular_bins: int = 72
    #: Soft inner radial cut: drop ``r < quantile(r, q)`` in plane. <0 disables.
    ring_inner_quantile: float = 0.30
    #: Upper radial outlier cut (``r > quantile(r, q)``). 1.0 disables.
    ring_outer_quantile: float = 0.995
    #: If False, do not clone E/W vertices into empty azimuth bins (honest coverage).
    angular_fill_empty_bins: bool = False
    #: Deprecated: global ``r >= frac * r_max`` drops ellipse N/S. <0 disables.
    ring_radius_min_frac: float = -1.0
    axis_hint: tuple[float, float, float] | None = None
    opening_end_mode: str = "away_from_point"
    opening_toward_env_point: tuple[float, float, float] = (0.14, 0.0, 0.84)
    opening_env_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    opening_env_axis_pick_positive: bool = True
    min_rim_points: int = 16
    vertices_frame: str = "env_local"
    env_origin: tuple[float, float, float] | None = None
    # Legacy (mouth_* from task cfg); mapped into plane/radial params when needed.
    entry_plane_ratio: float = 0.50
    ring_quantile: float = 0.65


@dataclass
class RimExtractResult:
    """Intermediate geometry for debug draw and PCA."""

    rim_points: torch.Tensor  # (N, 3)
    all_vertices: torch.Tensor  # (V, 3)
    plane_center: torch.Tensor  # (3,)
    plane_normal: torch.Tensor  # (3,) unit, opening normal ≈ cuff long axis
    plane_near_points: torch.Tensor  # (M, 3) within plane distance threshold
    radial_outer_points: torch.Tensor  # (R, 3) after in-plane radial filter, before angular bins
    plane_u: torch.Tensor  # (3,) in-plane basis
    plane_v: torch.Tensor  # (3,)
    plane_band_max: float = 0.0
    plane_distance_max: float = 0.0
    in_plane_center: torch.Tensor | None = None  # (3,) centroid of plane-near projections
    n_end_slice: int = 0
    n_plane_near: int = 0
    n_radial_outer: int = 0
    in_plane_radii: torch.Tensor | None = None  # (N,) radii of rim points
    angular_bins_filled: int = 0
    angular_bins_total: int = 0
    quadrant_counts: dict[str, int] = field(default_factory=dict)
    rim_point_global_indices: torch.Tensor | None = None  # (N,) global mesh indices
    #: Fitted ellipse loop (reference viz; measured rim may be sparser on hex meshes).
    rim_loop_viz: torch.Tensor | None = None


@dataclass
class PcaFrozenDebugComparison:
    """Side-by-side report: production ``pca_frozen`` vs geometric debug PCA."""

    lines: list[str]
    max_nsew_delta_mm: float = 0.0
    center_delta_mm: float = 0.0
    prod_index_in_debug_rim: dict[str, bool] = field(default_factory=dict)


@dataclass
class PCAResult:
    center: torch.Tensor
    eigenvalues: torch.Tensor  # shape (3,), descending λ1 ≥ λ2 ≥ λ3
    pc1: torch.Tensor
    pc2: torch.Tensor
    pc3: torch.Tensor
    axis_interpretation: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    eigenvalue_ratio_12: float = 0.0
    nsew_candidates: dict[str, torch.Tensor] = field(default_factory=dict)
    rim_extract: RimExtractResult | None = None

    def summary(self) -> str:
        lines = [
            "=== Bracelet opening PCA ===",
            f"center: {self.center.detach().cpu().tolist()}",
            f"eigenvalues (desc): {self.eigenvalues.detach().cpu().tolist()}",
            f"λ1/λ2 ratio: {self.eigenvalue_ratio_12:.4f}",
            f"pc1 (widest in-plane): {self.pc1.detach().cpu().tolist()} -> {self.axis_interpretation.get('pc1', '?')}",
            f"pc2 (narrow in-plane): {self.pc2.detach().cpu().tolist()} -> {self.axis_interpretation.get('pc2', '?')}",
            f"pc3 (thin/normal):     {self.pc3.detach().cpu().tolist()} -> {self.axis_interpretation.get('pc3', '?')}",
        ]
        if self.rim_extract is not None:
            re = self.rim_extract
            lines.append(
                f"rim extract: end_slice={re.n_end_slice}, plane_near={re.n_plane_near}, "
                f"radial_outer={re.n_radial_outer}, rim={re.rim_points.shape[0]}, "
                f"plane_band_max={re.plane_band_max:.5f}"
            )
            if re.quadrant_counts:
                qc = re.quadrant_counts
                lines.append(
                    f"rim quadrants: E={qc.get('east', 0)} N={qc.get('north', 0)} "
                    f"W={qc.get('west', 0)} S={qc.get('south', 0)}"
                )
            if re.in_plane_radii is not None and re.in_plane_radii.numel() > 0:
                r = re.in_plane_radii
                lines.append(
                    f"rim in-plane radius: min={float(r.min()):.4f} med={float(r.median()):.4f} max={float(r.max()):.4f}"
                )
            if re.angular_bins_total > 0:
                lines.append(
                    f"angular bins filled: {re.angular_bins_filled}/{re.angular_bins_total}"
                )
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        for name, pt in self.nsew_candidates.items():
            lines.append(f"  {name}: {pt.detach().cpu().tolist()}")
        return "\n".join(lines)


@dataclass
class OpeningPCAVisualization:
    """Cached state for per-frame debug redraw (env_id=0 only)."""

    env_id: int
    pca: PCAResult
    rim_points: torch.Tensor
    arrow_scale: float
    frame: str = "env_local"
    rim_extract: RimExtractResult | None = None
    rim_loop_viz: torch.Tensor | None = None
    production_nsew: dict[str, torch.Tensor] | None = None


def get_bracelet_vertices(
    env: Any,
    env_id: int = 0,
    *,
    use_rest_pose: bool = False,
    frame: str = "env_local",
) -> torch.Tensor:
    """Read deformable bracelet nodal positions from an Isaac Lab env."""
    obj = getattr(env, "object", None)
    if obj is None:
        raise AttributeError(
            "get_bracelet_vertices: env.object is missing. "
            "TODO: point to your DeformableObject (see isaaclab.assets.DeformableObject)."
        )
    data = obj.data
    if use_rest_pose:
        pos_w = data.default_nodal_state_w[env_id, :, :3].clone()
    else:
        pos_w = data.nodal_pos_w[env_id, :, :3].clone()

    pos_w = pos_w.to(device=env.device, dtype=torch.float32)
    if frame == "world":
        return pos_w
    if frame == "env_local":
        origin = env.scene.env_origins[env_id].to(device=pos_w.device, dtype=pos_w.dtype)
        return pos_w - origin.unsqueeze(0)
    raise ValueError(f"frame must be 'world' or 'env_local', got {frame!r}")


def _estimate_cuff_axis(p: torch.Tensor, axis_hint: tuple[float, float, float] | None) -> torch.Tensor:
    if axis_hint is None:
        x = p - p.mean(dim=0, keepdim=True)
        _, _, vt = torch.pca_lowrank(x, q=3, center=False)
        a = vt[:, 0]
    else:
        a = torch.tensor(axis_hint, device=p.device, dtype=p.dtype)
    return a / (a.norm() + 1e-8)


def _mouth_entry_mask(
    p: torch.Tensor,
    mask_plus: torch.Tensor,
    mask_minus: torch.Tensor,
    r: torch.Tensor,
    cfg: OpeningRimExtractConfig,
) -> torch.Tensor:
    """Pick hand-side vs grip-side end slice (env-local semantics)."""
    if mask_plus.sum() == 0:
        return mask_minus
    if mask_minus.sum() == 0:
        return mask_plus

    device, dtype = p.device, p.dtype
    mode = str(cfg.opening_end_mode).lower()
    r_plus, r_minus = r[mask_plus], r[mask_minus]

    if mode == "env_axis":
        axis = torch.tensor(cfg.opening_env_axis, device=device, dtype=dtype)
        axis = axis / (axis.norm() + 1e-8)
        pick_pos = bool(cfg.opening_env_axis_pick_positive)
        s_plus = (p[mask_plus] @ axis).mean()
        s_minus = (p[mask_minus] @ axis).mean()
        use_plus = s_plus >= s_minus if pick_pos else s_plus < s_minus
        return mask_plus if use_plus else mask_minus

    if mode in ("toward_point", "away_from_point"):
        if cfg.env_origin is not None:
            origin = torch.tensor(cfg.env_origin, device=device, dtype=dtype)
        else:
            origin = torch.zeros(3, device=device, dtype=dtype)
        hint = origin + torch.tensor(cfg.opening_toward_env_point, device=device, dtype=dtype)
        d_plus = torch.norm(p[mask_plus].mean(0) - hint)
        d_minus = torch.norm(p[mask_minus].mean(0) - hint)
        closer_plus = d_plus <= d_minus
        return mask_plus if (closer_plus if mode == "toward_point" else not closer_plus) else mask_minus

    if mode == "smaller_radius" and r_plus.numel() and r_minus.numel():
        return mask_plus if r_plus.median() < r_minus.median() else mask_minus
    if r_plus.numel() and r_minus.numel():
        return mask_plus if r_plus.median() >= r_minus.median() else mask_minus
    return mask_plus


def _plane_basis(n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal ``u, v`` spanning the plane with normal ``n``."""
    n = n / (torch.norm(n) + 1e-8)
    g = torch.tensor([0.0, 0.0, 1.0], device=n.device, dtype=n.dtype)
    if torch.abs(torch.dot(g, n)) > 0.9:
        g = torch.tensor([0.0, 1.0, 0.0], device=n.device, dtype=n.dtype)
    u = torch.linalg.cross(g, n)
    u = u / (torch.norm(u) + 1e-8)
    v = torch.linalg.cross(n, u)
    v = v / (torch.norm(v) + 1e-8)
    return u, v


def _project_to_plane(p: torch.Tensor, p0: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    d = ((p - p0.unsqueeze(0)) * n.unsqueeze(0)).sum(dim=1, keepdim=True)
    return p - d * n.unsqueeze(0)


def _ellipse_semi_axes(
    center: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    points: torch.Tensor,
) -> tuple[float, float]:
    rel = points - center.unsqueeze(0)
    semi_u = float((rel * u.unsqueeze(0)).sum(-1).abs().max().item())
    semi_v = float((rel * v.unsqueeze(0)).sum(-1).abs().max().item())
    return max(semi_u, 1e-6), max(semi_v, 1e-6)


def _synthesize_ellipse_rim(
    center: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    semi_u: float,
    semi_v: float,
    n_samples: int = 72,
) -> torch.Tensor:
    """Synthetic full ring in the opening plane (viz only; not mesh vertex indices)."""
    device, dtype = center.device, center.dtype
    t = torch.linspace(-math.pi, math.pi, int(n_samples) + 1, device=device, dtype=dtype)[:-1]
    return (
        center.unsqueeze(0)
        + semi_u * torch.cos(t).unsqueeze(1) * u.unsqueeze(0)
        + semi_v * torch.sin(t).unsqueeze(1) * v.unsqueeze(0)
    )


def _angular_outer_pick(
    r_ip: torch.Tensor,
    angles: torch.Tensor,
    n_bins: int,
    *,
    fill_empty_bins: bool = False,
) -> tuple[torch.Tensor, int]:
    """Per azimuth bin, index of the outermost point (max in-plane radius).

    Returns (local_indices, n_bins_filled).
    """
    if r_ip.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=r_ip.device), 0
    two_pi = 2.0 * math.pi
    bin_id = torch.clamp(((angles + math.pi) / two_pi * float(n_bins)).long(), 0, n_bins - 1)
    bin_best: dict[int, int] = {}
    for b in range(n_bins):
        m = bin_id == b
        if not m.any():
            continue
        local = torch.nonzero(m, as_tuple=False).squeeze(1)
        best = local[torch.argmax(r_ip[local])]
        bin_best[b] = int(best.item())

    if not bin_best:
        return torch.arange(r_ip.shape[0], device=r_ip.device, dtype=torch.long), 0

    if fill_empty_bins:
        filled = sorted(bin_best.keys())
        for b in range(n_bins):
            if b in bin_best:
                continue
            nearest = min(filled, key=lambda fb: min((b - fb) % n_bins, (fb - b) % n_bins))
            bin_best[b] = bin_best[nearest]

    picks = torch.tensor(sorted(set(bin_best.values())), device=r_ip.device, dtype=torch.long)
    return picks, len(bin_best)


def _quadrant_counts_from_angles(angles: torch.Tensor) -> dict[str, int]:
    """Count rim samples per 90° sector in the opening plane (E/N/W/S)."""
    quarter = math.pi / 4.0
    counts = {"east": 0, "north": 0, "west": 0, "south": 0}
    for i in range(int(angles.numel())):
        ang = float(angles[i].item())
        if -quarter <= ang < quarter:
            counts["east"] += 1
        elif quarter <= ang < 3.0 * quarter:
            counts["north"] += 1
        elif ang >= 3.0 * quarter or ang < -3.0 * quarter:
            counts["west"] += 1
        else:
            counts["south"] += 1
    return counts


def extract_opening_rim_points_geometric(
    vertices: torch.Tensor,
    config: OpeningRimExtractConfig | None = None,
    *,
    return_debug: bool = False,
) -> torch.Tensor | RimExtractResult:
    """Extract opening-rim points: plane distance → in-plane radial outer → angular outer pick.

    Previous failure mode (left/right strips only)
    ----------------------------------------------
    - ``mask_t_side`` (outer 45% along cuff) dropped N/S cap vertices slightly inboard on ``a``.
    - ``ring_radius_min_frac * r_max`` from task ``mouth_ring_quantile`` (~0.71) removed ellipse
      N/S (short axis ~0.35× long axis). Only E/W wall strips survived.
    """
    cfg = config or OpeningRimExtractConfig()
    p = vertices.to(dtype=torch.float32)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"vertices must be (V, 3), got {tuple(p.shape)}")
    if p.shape[0] < cfg.min_rim_points:
        raise RuntimeError(f"Too few vertices: {p.shape[0]} < min_rim_points={cfg.min_rim_points}")

    mean = p.mean(dim=0)
    x = p - mean
    a = _estimate_cuff_axis(p, cfg.axis_hint)

    t = x @ a
    t_min, t_max = t.min(), t.max()
    cuff_len = float((t_max - t_min).item())
    sl_probe = float(cfg.end_slice_ratio) * cuff_len
    sl_expanded = float(cfg.end_slice_ratio) * float(cfg.end_cap_expand) * cuff_len
    mask_plus = t >= (t_max - sl_probe)
    mask_minus = t <= (t_min + sl_probe)
    x_perp = x - t.unsqueeze(1) * a.unsqueeze(0)
    r_axis = torch.linalg.norm(x_perp, dim=1)

    mask_entry = _mouth_entry_mask(p, mask_plus, mask_minus, r_axis, cfg)
    idx_end = torch.nonzero(mask_entry, as_tuple=False).squeeze(1)
    if idx_end.numel() == 0:
        raise RuntimeError(
            "Opening end slice empty; tune end_slice_ratio / opening_end_mode / opening_toward_env_point."
        )
    on_plus_end = bool(mask_plus[idx_end].float().mean().item() > 0.5)
    p0 = p[idx_end].mean(dim=0)
    n = a if on_plus_end else -a
    n = n / (n.norm() + 1e-8)
    u, v = _plane_basis(n)

    if on_plus_end:
        mask_end_expanded = t >= (t_max - sl_expanded)
    else:
        mask_end_expanded = t <= (t_min + sl_expanded)
    idx_end_exp = torch.nonzero(mask_end_expanded, as_tuple=False).squeeze(1)
    if idx_end_exp.numel() == 0:
        idx_end_exp = idx_end

    plane_d_all = ((p - p0.unsqueeze(0)) * n.unsqueeze(0)).sum(dim=1).abs()
    plane_band_max = max(
        sl_expanded * float(cfg.plane_band_scale),
        float(cfg.plane_band_scale)
        * float(torch.quantile(plane_d_all[idx_end_exp], float(cfg.plane_band_quantile)).item()),
    )
    mask_plane_near = mask_end_expanded & (plane_d_all <= plane_band_max)
    idx_plane_near = torch.nonzero(mask_plane_near, as_tuple=False).squeeze(1)
    if idx_plane_near.numel() < cfg.min_rim_points:
        idx_plane_near = idx_end_exp

    p_near = p[idx_plane_near]
    p_proj = _project_to_plane(p_near, p0, n)
    c = p_proj.mean(dim=0)
    rel = p_proj - c.unsqueeze(0)
    r_ip = torch.linalg.norm(rel, dim=1)
    angles = torch.atan2((rel * v.unsqueeze(0)).sum(dim=1), (rel * u.unsqueeze(0)).sum(dim=1))

    radial_mask = torch.ones(r_ip.shape[0], dtype=torch.bool, device=r_ip.device)
    inner_q = float(cfg.ring_inner_quantile)
    if inner_q >= 0.0:
        radial_mask = radial_mask & (r_ip >= torch.quantile(r_ip, inner_q))
    outer_q = float(cfg.ring_outer_quantile)
    if 0.0 < outer_q < 1.0:
        radial_mask = radial_mask & (r_ip <= torch.quantile(r_ip, outer_q))
    min_frac = float(cfg.ring_radius_min_frac)
    if min_frac >= 0.0:
        r_max = float(r_ip.max().item())
        radial_mask = radial_mask & (r_ip >= min_frac * r_max)
    if int(radial_mask.sum()) < cfg.min_rim_points:
        radial_mask = torch.ones_like(radial_mask)

    pool_local = torch.nonzero(radial_mask, as_tuple=False).squeeze(1)
    radial_outer_points = p_near[pool_local]
    n_bins = max(16, int(cfg.angular_bins))
    fill_bins = bool(cfg.angular_fill_empty_bins)

    def _pick(pool_idx: torch.Tensor) -> tuple[torch.Tensor, int]:
        return _angular_outer_pick(
            r_ip[pool_idx], angles[pool_idx], n_bins, fill_empty_bins=fill_bins
        )

    pick_local, n_filled = _pick(pool_local)
    keep = idx_plane_near[pool_local[pick_local]]

    if keep.numel() < cfg.min_rim_points or n_filled < max(12, n_bins // 3):
        pick_local, n_filled = _pick(torch.arange(r_ip.shape[0], device=r_ip.device, dtype=torch.long))
        keep = idx_plane_near[pick_local]

    rim_points = p[keep]
    rim_proj = _project_to_plane(rim_points, c, n)
    rim_radii = torch.linalg.norm(rim_proj - c.unsqueeze(0), dim=1)
    rel_rim = rim_proj - c.unsqueeze(0)
    rim_angles = torch.atan2(
        (rel_rim * v.unsqueeze(0)).sum(-1),
        (rel_rim * u.unsqueeze(0)).sum(-1),
    )

    result = RimExtractResult(
        rim_points=rim_points,
        all_vertices=p,
        plane_center=p0,
        plane_normal=n,
        plane_near_points=p_near,
        radial_outer_points=radial_outer_points,
        plane_u=u,
        plane_v=v,
        plane_band_max=float(plane_band_max),
        plane_distance_max=float(plane_band_max),
        in_plane_center=c,
        n_end_slice=int(idx_end.numel()),
        n_plane_near=int(idx_plane_near.numel()),
        n_radial_outer=int(radial_outer_points.shape[0]),
        in_plane_radii=rim_radii,
        angular_bins_filled=n_filled,
        angular_bins_total=n_bins,
        quadrant_counts=_quadrant_counts_from_angles(rim_angles),
        rim_point_global_indices=keep.long(),
    )

    if return_debug:
        return result
    return rim_points


def compute_pca(points: torch.Tensor) -> PCAResult:
    """PCA on rim point cloud; eigenvalues sorted **descending**."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {tuple(points.shape)}")
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points for PCA")

    center = points.mean(dim=0)
    x = points - center.unsqueeze(0)
    n_pts = int(x.shape[0])
    cov = (x.T @ x) / max(n_pts, 1)
    evals_asc, evecs_asc = torch.linalg.eigh(cov)

    order = torch.argsort(evals_asc, descending=True)
    eigenvalues = evals_asc[order]
    evecs = evecs_asc[:, order]
    pc1 = evecs[:, 0] / (torch.norm(evecs[:, 0]) + 1e-8)
    pc2 = evecs[:, 1] / (torch.norm(evecs[:, 1]) + 1e-8)
    pc3 = evecs[:, 2] / (torch.norm(evecs[:, 2]) + 1e-8)

    up = torch.tensor([0.0, 0.0, 1.0], device=points.device, dtype=points.dtype)
    if torch.dot(pc3, up) < 0:
        pc3 = -pc3
    pc2 = torch.linalg.cross(pc3, pc1)
    pc2 = pc2 / (torch.norm(pc2) + 1e-8)
    if torch.dot(pc2, up) < 0:
        pc2 = -pc2
        pc1 = torch.linalg.cross(pc2, pc3)
        pc1 = pc1 / (torch.norm(pc1) + 1e-8)

    result = PCAResult(
        center=center,
        eigenvalues=eigenvalues,
        pc1=pc1,
        pc2=pc2,
        pc3=pc3,
    )
    result.eigenvalue_ratio_12 = float((eigenvalues[0] / (eigenvalues[1] + 1e-12)).item())
    if result.eigenvalue_ratio_12 < 1.25:
        result.warnings.append(
            f"λ1/λ2={result.eigenvalue_ratio_12:.3f} < 1.25: opening is nearly circular; "
            "in-plane PCA axes (pc1/pc2) are poorly conditioned."
        )
    if float(eigenvalues[2] / (eigenvalues[0] + 1e-12)) > 0.35:
        result.warnings.append(
            "λ3 is relatively large vs λ1: rim may not be planar (heavy distortion or bad rim extraction)."
        )
    return result


def interpret_pca_axes(
    pc1: torch.Tensor,
    pc2: torch.Tensor,
    pc3: torch.Tensor,
    *,
    frame_label: str = "env-local",
) -> dict[str, str]:
    """Map each PC to the closest world/env X/Y/Z axis by absolute dot product."""

    def _closest(v: torch.Tensor) -> str:
        v = v / (torch.norm(v) + 1e-8)
        best_i, best_dot = 0, -1.0
        for i, axis in enumerate(WORLD_AXIS_VECTORS):
            d = abs(float(torch.dot(v, torch.tensor(axis, device=v.device, dtype=v.dtype)).item()))
            if d > best_dot:
                best_dot, best_i = d, i
        return f"{frame_label} {WORLD_AXIS_NAMES[best_i]} (|dot|={best_dot:.3f})"

    return {"pc1": _closest(pc1), "pc2": _closest(pc2), "pc3": _closest(pc3)}


def _opening_plane_axes_from_pca(pca: PCAResult) -> tuple[torch.Tensor, torch.Tensor]:
    """In-plane ``u`` (≈ pc1 / wide) and ``v`` (≈ pc2 / narrow), orthonormal."""
    u = pca.pc1 / (torch.norm(pca.pc1) + 1e-8)
    v = pca.pc2 / (torch.norm(pca.pc2) + 1e-8)
    v = v - torch.dot(v, u) * u
    v = v / (torch.norm(v) + 1e-8)
    if torch.dot(v, pca.pc2) < 0:
        v = -v
    return u, v


def _angle_in_sector(ang: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Angular mask on ``[-pi, pi]``; supports west wrap across ±pi."""
    if lo <= hi:
        return (ang >= lo) & (ang < hi)
    return (ang >= lo) | (ang < hi)


def _pick_rim_point_on_vertical_axis(
    rim_points: torch.Tensor,
    center: torch.Tensor,
    *,
    upward: bool,
) -> torch.Tensor:
    """Rim vertex closest to the env-local vertical line through ``center`` (+Z or −Z).

    Minimizes lateral offset ``x² + y²`` from the center (not global max/min Z).
    """
    rel = rim_points - center.unsqueeze(0)
    lateral = rel[:, 0] ** 2 + rel[:, 1] ** 2
    if upward:
        mask = rel[:, 2] >= 0.0
    else:
        mask = rel[:, 2] <= 0.0
    if not mask.any():
        mask = torch.ones_like(mask, dtype=torch.bool)
    scores = torch.where(mask, lateral, torch.full_like(lateral, float("inf")))
    return rim_points[torch.argmin(scores)]


def _east_west_from_plane_sectors(
    rim_points: torch.Tensor,
    center: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Pick outermost rim point in each E/W 90° sector of the opening plane."""
    rel = rim_points - center.unsqueeze(0)
    pu = (rel * u.unsqueeze(0)).sum(-1)
    pv = (rel * v.unsqueeze(0)).sum(-1)
    ang = torch.atan2(pv, pu)
    rad = torch.linalg.norm(rel, dim=1)
    quarter = math.pi / 4.0
    sectors: dict[str, tuple[float, float]] = {
        "east": (-quarter, quarter),
        "west": (3.0 * quarter, math.pi),
    }
    out: dict[str, torch.Tensor] = {}
    for name, (lo, hi) in sectors.items():
        m = _angle_in_sector(ang, lo, hi)
        if name == "west":
            m = m | _angle_in_sector(ang, -math.pi, -3.0 * quarter)
        if not m.any():
            mid = 0.5 * (lo + hi)
            m = torch.abs(ang - mid) == torch.abs(ang - mid).min()
        idx = torch.nonzero(m, as_tuple=False).squeeze(1)
        best = idx[torch.argmax(rad[idx])]
        out[name] = rim_points[best]
    return out


def _nsew_candidates_from_pca(
    rim_points: torch.Tensor,
    pca: PCAResult,
) -> dict[str, torch.Tensor]:
    """E/W via plane sectors; N/S via vertical line through PCA center (+Z / −Z)."""
    u, v = _opening_plane_axes_from_pca(pca)
    out = _east_west_from_plane_sectors(rim_points, pca.center, u, v)
    out["north"] = _pick_rim_point_on_vertical_axis(rim_points, pca.center, upward=True)
    out["south"] = _pick_rim_point_on_vertical_axis(rim_points, pca.center, upward=False)
    return out


def _should_use_opening_ring_for_debug(env: Any) -> bool:
    """Use production ``opening_ring`` pool when task runs ``pca_frozen`` / ``pca``."""
    cfg = getattr(env, "cfg", None)
    if cfg is None or not bool(getattr(cfg, "debug_opening_pca_use_opening_ring", True)):
        return False
    rim_idx = getattr(env, "_bracelet_rim_idx", None)
    if rim_idx is None or int(rim_idx.numel()) < 3:
        return False
    mode = str(getattr(cfg, "deformable_bracelet_nsew_geom_mode", "")).lower()
    return mode in ("pca_frozen", "pca")


def _rim_row_for_global_idx(p_row: torch.Tensor, rim_idx: torch.Tensor, global_idx: int) -> torch.Tensor:
    local = int(torch.nonzero(rim_idx == int(global_idx), as_tuple=False).squeeze(1)[0].item())
    return p_row[local]


def _nsew_pca_frozen_on_opening_ring(
    env: Any,
    rim_points: torch.Tensor,
    rim_idx: torch.Tensor,
    e1: torch.Tensor,
    e2: torch.Tensor,
    x_centered: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """NSEW positions using production ``pca_frozen`` rules on ``opening_ring``."""
    mouths = env._pick_nsew_global_indices_pca_frozen_ring(
        rim_points, rim_idx, e1, e2, x_centered
    )
    return {
        name: _rim_row_for_global_idx(rim_points, rim_idx, mouths[name])
        for name in ("east", "west", "north", "south")
    }


def compute_opening_ring_pca_debug(
    env: Any,
    vertices: torch.Tensor,
    env_id: int = 0,
    *,
    frame: str = "env_local",
) -> tuple[PCAResult, RimExtractResult]:
    """Debug PCA on production ``opening_ring`` with ``pca_frozen`` axes and NSEW rules."""
    if not hasattr(env, "_compute_rest_rim_pca_frame"):
        raise RuntimeError("compute_opening_ring_pca_debug requires env._compute_rest_rim_pca_frame")
    rim_idx = env._bracelet_rim_idx.to(device=vertices.device, dtype=torch.long)
    rim_points = vertices[rim_idx]
    r = env._compute_rest_rim_pca_frame()
    if r is None:
        raise RuntimeError("opening_ring PCA frame unavailable (check _bracelet_geom_e1_ref)")
    _, e1, e2, x = r
    n = torch.linalg.cross(e1, e2)
    n = n / (torch.norm(n) + 1e-8)
    center = rim_points.mean(dim=0)

    pca_cov = compute_pca(rim_points)
    pca = PCAResult(
        center=center,
        eigenvalues=pca_cov.eigenvalues,
        pc1=e1.clone(),
        pc2=e2.clone(),
        pc3=n.clone(),
        eigenvalue_ratio_12=pca_cov.eigenvalue_ratio_12,
        warnings=list(pca_cov.warnings),
    )
    pca.axis_interpretation = interpret_pca_axes(pca.pc1, pca.pc2, pca.pc3, frame_label=frame)

    snap = env.get_pca_frozen_nsew_snapshot(env_id) if hasattr(env, "get_pca_frozen_nsew_snapshot") else None
    if snap is not None:
        pca.nsew_candidates = {k: v.clone() for k, v in snap["positions_env_local"].items()}
    else:
        pca.nsew_candidates = _nsew_pca_frozen_on_opening_ring(env, rim_points, rim_idx, e1, e2, x)

    rel = rim_points - center.unsqueeze(0)
    angles = torch.atan2((rel * e2.unsqueeze(0)).sum(-1), (rel * e1.unsqueeze(0)).sum(-1))
    n_ring = int(rim_idx.numel())
    rim_radii = torch.linalg.norm(rel, dim=1)
    rim_debug = RimExtractResult(
        rim_points=rim_points,
        all_vertices=vertices,
        plane_center=center,
        plane_normal=n,
        plane_near_points=rim_points,
        radial_outer_points=rim_points,
        plane_u=e1,
        plane_v=e2,
        plane_band_max=0.0,
        plane_distance_max=0.0,
        in_plane_center=center,
        n_end_slice=n_ring,
        n_plane_near=n_ring,
        n_radial_outer=n_ring,
        in_plane_radii=rim_radii,
        angular_bins_filled=n_ring,
        angular_bins_total=n_ring,
        quadrant_counts=_quadrant_counts_from_angles(angles),
        rim_point_global_indices=rim_idx,
    )
    semi_u, semi_v = _ellipse_semi_axes(center, e1, e2, rim_points)
    rim_debug.rim_loop_viz = _synthesize_ellipse_rim(
        center, e1, e2, semi_u, semi_v, n_samples=max(72, n_ring)
    )
    return pca, rim_debug


def _acquire_debug_draw_interface():
    try:
        import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

        return omni_debug_draw.acquire_debug_draw_interface()
    except ImportError:
        pass
    try:
        import omni.isaac.debug_draw._debug_draw as omni_debug_draw

        return omni_debug_draw.acquire_debug_draw_interface()
    except ImportError:
        logger.warning("debug_draw unavailable (headless or missing isaacsim.util.debug_draw).")
        return None


def _subsample_points(p: torch.Tensor, max_n: int) -> torch.Tensor:
    if p.shape[0] <= max_n:
        return p
    step = max(1, p.shape[0] // max_n)
    return p[::step]


def _draw_plane_grid(
    iface: Any,
    p0: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    half_extent: float,
    n_grid: int,
    to_world,
) -> None:
    """Draw opening-plane candidate as a gray grid."""
    color = (0.55, 0.55, 0.55, 0.45)
    for i in range(-n_grid, n_grid + 1):
        alpha = i / max(n_grid, 1) * half_extent
        pa = p0 + u * alpha - v * half_extent
        pb = p0 + u * alpha + v * half_extent
        iface.draw_lines([to_world(pa)], [to_world(pb)], [list(color)], [1.0])
        pa = p0 + v * alpha - u * half_extent
        pb = p0 + v * alpha + u * half_extent
        iface.draw_lines([to_world(pa)], [to_world(pb)], [list(color)], [1.0])


def _draw_rim_loop(
    iface: Any,
    rim_points: torch.Tensor,
    center: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    to_world,
    *,
    color: tuple[float, float, float, float] = (1.0, 0.55, 0.1, 0.9),
    line_width: float = 2.5,
) -> None:
    """Connect rim points by polar angle in the opening plane."""
    if rim_points.shape[0] < 2:
        return
    rel = rim_points - center.unsqueeze(0)
    ang = torch.atan2((rel * v).sum(-1), (rel * u).sum(-1))
    order = torch.argsort(ang)
    rp = rim_points[order]
    for i in range(rp.shape[0]):
        j = (i + 1) % rp.shape[0]
        iface.draw_lines([to_world(rp[i])], [to_world(rp[j])], [list(color)], [line_width])


def _draw_cardinal_cross(
    iface: Any,
    center: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    extent: float,
    to_world,
) -> None:
    """E/W and N/S reference lines in the opening plane."""
    color = (0.45, 0.45, 0.45, 0.55)
    iface.draw_lines(
        [to_world(center - u * extent)],
        [to_world(center + u * extent)],
        [list(color)],
        [1.0],
    )
    iface.draw_lines(
        [to_world(center - v * extent)],
        [to_world(center + v * extent)],
        [list(color)],
        [1.0],
    )


def compare_debug_pca_with_pca_frozen(
    env: Any,
    debug_pca: PCAResult,
    env_id: int = 0,
) -> PcaFrozenDebugComparison | None:
    """Compare production ``pca_frozen`` NSEW/center with geometric debug PCA (env-local)."""
    if not hasattr(env, "get_pca_frozen_nsew_snapshot"):
        return None
    prod = env.get_pca_frozen_nsew_snapshot(env_id)
    if prod is None:
        return None

    lines = ["=== pca_frozen (production) vs debug PCA ==="]
    nsew_mode = str(getattr(getattr(env, "cfg", None), "deformable_bracelet_nsew_geom_mode", "")).lower()
    lines.append(f"nsew_mode={nsew_mode!r}  rim_vertex_set={prod.get('rim_vertex_set', '?')!r}")

    re = debug_pca.rim_extract
    n_debug_rim = int(re.rim_points.shape[0]) if re is not None else 0
    n_prod_ring = int(prod.get("rim_vertex_count", 0))
    debug_pool_label = "opening_ring" if n_prod_ring == n_debug_rim and n_prod_ring > 0 else "geometric rim"
    lines.append(f"rim pools: production opening_ring={n_prod_ring}, debug {debug_pool_label}={n_debug_rim}")

    rim_idx_set: set[int] = set()
    if re is not None and re.rim_point_global_indices is not None:
        rim_idx_set = {int(i) for i in re.rim_point_global_indices.detach().cpu().tolist()}

    opening_ring_set: set[int] = set()
    rim_idx = getattr(env, "_bracelet_rim_idx", None)
    if rim_idx is not None:
        opening_ring_set = {int(i) for i in rim_idx.detach().cpu().tolist()}
    if opening_ring_set and rim_idx_set:
        overlap = len(opening_ring_set & rim_idx_set)
        lines.append(f"opening_ring ∩ debug_rim_indices: {overlap}/{n_prod_ring}")

    max_delta = 0.0
    prod_in_debug: dict[str, bool] = {}
    for name in ("north", "south", "east", "west"):
        prod_pos = prod["positions_env_local"][name]
        debug_pos = debug_pca.nsew_candidates.get(name)
        prod_idx = int(prod["indices"][name])
        in_rim = prod_idx in rim_idx_set
        prod_in_debug[name] = in_rim
        if debug_pos is None:
            lines.append(f"  {name}: debug missing")
            continue
        delta_mm = float(torch.norm(prod_pos - debug_pos.to(prod_pos.device)).item()) * 1000.0
        max_delta = max(max_delta, delta_mm)
        lines.append(
            f"  {name}: Δ={delta_mm:.2f} mm  prod_idx={prod_idx}  in_debug_rim={in_rim}"
        )
        lines.append(f"    prod:  {[round(float(x), 5) for x in prod_pos.detach().cpu().tolist()]}")
        lines.append(f"    debug: {[round(float(x), 5) for x in debug_pos.detach().cpu().tolist()]}")

    center_delta_mm = 0.0
    prod_center = prod.get("center")
    if prod_center is not None:
        center_delta_mm = float(torch.norm(prod_center - debug_pca.center.to(prod_center.device)).item()) * 1000.0
        lines.append(
            f"center Δ={center_delta_mm:.2f} mm  "
            f"prod={[round(float(x), 5) for x in prod_center.detach().cpu().tolist()]}  "
            f"debug={[round(float(x), 5) for x in debug_pca.center.detach().cpu().tolist()]}"
        )

    unified = (
        n_prod_ring > 0
        and n_prod_ring == n_debug_rim
        and opening_ring_set
        and rim_idx_set
        and len(opening_ring_set & rim_idx_set) == n_prod_ring
        and max_delta < 0.05
        and center_delta_mm < 0.05
    )
    if unified:
        lines.append(
            "rules: UNIFIED — debug uses production opening_ring, e1 extreme-band (E/W), "
            "Z-axis (N/S), frozen pca_frozen snapshot"
        )
        lines.append("status: production and debug match (Δ < 0.05 mm)")
    else:
        lines.append(
            "rules: prod E/W = e1 extreme-band on opening_ring; "
            "debug E/W = plane-sector on geometric rim"
        )
        lines.append(
            "       prod N/S = Z-axis through opening_ring centroid; "
            "debug N/S = Z-axis through debug PCA center"
        )
        if max_delta > 5.0 or center_delta_mm > 3.0:
            lines.append(
                "NOTE: large Δ expected when rim pools differ (opening_ring ~72 vs geometric ~38)."
            )
    return PcaFrozenDebugComparison(
        lines=lines,
        max_nsew_delta_mm=max_delta,
        center_delta_mm=center_delta_mm,
        prod_index_in_debug_rim=prod_in_debug,
    )


def draw_opening_pca_debug(
    center: torch.Tensor,
    pc1: torch.Tensor,
    pc2: torch.Tensor,
    pc3: torch.Tensor,
    rim_points: torch.Tensor,
    scale: float = 0.08,
    *,
    env_origin: torch.Tensor | None = None,
    nsew_candidates: dict[str, torch.Tensor] | None = None,
    production_nsew: dict[str, torch.Tensor] | None = None,
    rim_extract: RimExtractResult | None = None,
    draw_interface: Any | None = None,
    max_rim_dots: int = 400,
    max_all_dots: int = 800,
) -> Any | None:
    """Debug draw: full pipeline layers + PCA + NSEW.

    Colors
    ------
    - all mesh vertices: dark gray (small)
    - opening plane grid + normal: light gray
    - plane-near (|dot(p-p0,n)| band): magenta
    - radial outer annulus (pre-angular): yellow-green
    - rim points: cyan (large)
    - measured rim loop: orange solid
    - fitted ellipse reference: orange dashed (low alpha)
    - plane cardinal cross: gray
    - PCA center: white; pc1/pc2/pc3: red / green / blue
    - N/S/E/W candidates (debug): colored ticks
    - N/S/E/W (production pca_frozen): white squares + thin ticks
    """
    iface = draw_interface if draw_interface is not None else _acquire_debug_draw_interface()
    if iface is None:
        return None

    device = center.device
    dtype = center.dtype
    o = torch.zeros(3, device=device, dtype=dtype) if env_origin is None else env_origin.to(device=device, dtype=dtype)

    def _w(p: torch.Tensor) -> list[float]:
        return (p + o).detach().cpu().tolist()

    c = center.detach()
    iface.clear_lines()
    iface.clear_points()

    loop_center = c
    plane_u = rim_extract.plane_u if rim_extract is not None else pc1
    plane_v = rim_extract.plane_v if rim_extract is not None else pc2

    if rim_extract is not None:
        all_v = rim_extract.all_vertices
        if all_v.shape[0] > max_all_dots:
            all_v = _subsample_points(all_v, max_all_dots)
        iface.draw_points(
            [_w(all_v[i]) for i in range(all_v.shape[0])],
            [(0.35, 0.35, 0.35, 0.35)] * all_v.shape[0],
            [2.0] * all_v.shape[0],
        )

        if rim_extract.in_plane_center is not None:
            loop_center = rim_extract.in_plane_center.detach()

        half = float(rim_extract.in_plane_radii.max().item()) * 1.2 if (
            rim_extract.in_plane_radii is not None and rim_extract.in_plane_radii.numel() > 0
        ) else scale * 1.5
        if rim_extract.plane_near_points.shape[0] > 0:
            pn_rel = _project_to_plane(
                rim_extract.plane_near_points, rim_extract.plane_center, rim_extract.plane_normal
            ) - loop_center.unsqueeze(0)
            half = max(half, float(torch.linalg.norm(pn_rel, dim=1).max().item()) * 1.1)
        _draw_plane_grid(
            iface, rim_extract.plane_center, rim_extract.plane_u, rim_extract.plane_v, half, 5, _w
        )
        _draw_cardinal_cross(iface, loop_center, plane_u, plane_v, half, _w)

        n_hat = rim_extract.plane_normal / (torch.norm(rim_extract.plane_normal) + 1e-8)
        p_plane_vis = rim_extract.plane_center
        iface.draw_lines(
            [_w(p_plane_vis)],
            [_w(p_plane_vis + n_hat * scale * 0.5)],
            [(0.7, 0.7, 0.7, 0.8)],
            [2.0],
        )

        pn = rim_extract.plane_near_points
        if pn.shape[0] > max_all_dots:
            pn = _subsample_points(pn, max_all_dots)
        iface.draw_points(
            [_w(pn[i]) for i in range(pn.shape[0])],
            [(0.9, 0.2, 0.9, 0.55)] * pn.shape[0],
            [3.0] * pn.shape[0],
        )

        ro = rim_extract.radial_outer_points
        if ro.shape[0] > 0:
            if ro.shape[0] > max_all_dots:
                ro = _subsample_points(ro, max_all_dots)
            iface.draw_points(
                [_w(ro[i]) for i in range(ro.shape[0])],
                [(0.75, 0.95, 0.2, 0.65)] * ro.shape[0],
                [3.5] * ro.shape[0],
            )

    rp = rim_points
    if rp.shape[0] > max_rim_dots:
        rp = _subsample_points(rp, max_rim_dots)
    iface.draw_points(
        [_w(rp[i]) for i in range(rp.shape[0])],
        [(0.2, 0.9, 1.0, 1.0)] * rp.shape[0],
        [5.0] * rp.shape[0],
    )

    if rim_extract is not None:
        _draw_rim_loop(
            iface, rim_points, loop_center, plane_u, plane_v, _w,
            color=(1.0, 0.55, 0.1, 0.95), line_width=3.0,
        )
        if rim_extract.rim_loop_viz is not None:
            _draw_rim_loop(
                iface, rim_extract.rim_loop_viz, loop_center, plane_u, plane_v, _w,
                color=(1.0, 0.75, 0.35, 0.45), line_width=1.5,
            )

    iface.draw_points([_w(c)], [(1.0, 1.0, 1.0, 1.0)], [8.0])

    def _arrow(axis: torch.Tensor, color: tuple[float, float, float, float]) -> None:
        p0 = _w(c)
        p1 = _w(c + scale * axis)
        iface.draw_lines([p0], [p1], [list(color)], [4.0])

    _arrow(pc1, (1.0, 0.2, 0.2, 1.0))
    _arrow(pc2, (0.2, 1.0, 0.2, 1.0))
    _arrow(pc3, (0.3, 0.5, 1.0, 1.0))
    for axis, color in ((pc1, (0.6, 0.1, 0.1, 0.8)), (pc2, (0.1, 0.6, 0.1, 0.8)), (pc3, (0.2, 0.3, 0.6, 0.8))):
        iface.draw_lines([_w(c)], [_w(c - 0.65 * scale * axis)], [list(color)], [2.0])

    if nsew_candidates:
        if rim_extract is not None:
            u_ax = rim_extract.plane_u / (torch.norm(rim_extract.plane_u) + 1e-8)
            v_ax = rim_extract.plane_v / (torch.norm(rim_extract.plane_v) + 1e-8)
        else:
            u_ax, v_ax = _opening_plane_axes_from_pca(
                PCAResult(
                    center=center,
                    eigenvalues=torch.zeros(3, device=device, dtype=dtype),
                    pc1=pc1,
                    pc2=pc2,
                    pc3=pc3,
                )
            )
        tick_len = max(scale * 0.35, 0.015)
        z_ax = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        nsew_style = {
            "east": ((1.0, 0.85, 0.0, 1.0), u_ax * tick_len),
            "west": ((1.0, 0.55, 0.0, 1.0), -u_ax * tick_len),
            "north": ((0.2, 1.0, 0.3, 1.0), z_ax * tick_len),
            "south": ((0.2, 0.7, 1.0, 1.0), -z_ax * tick_len),
        }
        for name, pt in nsew_candidates.items():
            p0 = _w(pt)
            color, off = nsew_style.get(name, ((1.0, 1.0, 0.0, 1.0), torch.zeros(3, device=device, dtype=dtype)))
            iface.draw_lines([p0], [_w(pt + off)], [list(color)], [5.0])
            iface.draw_points([p0], [list(color)], [10.0])

    if production_nsew:
        z_ax = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        prod_style = {
            "east": z_ax * 0.0 + torch.tensor([0.0, -1.0, 0.0], device=device, dtype=dtype),
            "west": z_ax * 0.0 + torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype),
            "north": z_ax,
            "south": -z_ax,
        }
        tick_len = max(scale * 0.28, 0.012)
        for name, pt in production_nsew.items():
            p0 = _w(pt)
            off_dir = prod_style.get(name, z_ax)
            iface.draw_lines([p0], [_w(pt + off_dir * tick_len)], [(1.0, 1.0, 1.0, 0.85)], [3.0])
            iface.draw_points([p0], [(1.0, 1.0, 1.0, 0.95)], [12.0])

    return iface


def refresh_opening_pca_debug_draw(env: Any) -> None:
    """Redraw cached PCA debug (call each simulation frame if using debug_draw)."""
    viz = getattr(env, "_opening_pca_viz", None)
    if viz is None:
        return
    origin = env.scene.env_origins[viz.env_id]
    draw_opening_pca_debug(
        viz.pca.center,
        viz.pca.pc1,
        viz.pca.pc2,
        viz.pca.pc3,
        viz.rim_points,
        scale=viz.arrow_scale,
        env_origin=origin if viz.frame == "env_local" else None,
        nsew_candidates=viz.pca.nsew_candidates,
        production_nsew=viz.production_nsew,
        rim_extract=viz.rim_extract,
        draw_interface=getattr(env, "_opening_pca_draw_iface", None),
    )


def compute_bracelet_opening_pca(
    env: Any,
    env_id: int = 0,
    *,
    use_rest_pose: bool = True,
    draw: bool = False,
    arrow_scale: float = 0.08,
    extract_config: OpeningRimExtractConfig | None = None,
    cache_on_env: bool = True,
) -> PCAResult:
    """Main entry: opening_ring (pca_frozen) or geometric rim → PCA → log (+ optional debug draw)."""
    frame = "env_local"
    vertices = get_bracelet_vertices(env, env_id, use_rest_pose=use_rest_pose, frame=frame)
    rim_source = "geometric"
    use_ring = _should_use_opening_ring_for_debug(env)

    if use_ring:
        rim_source = "opening_ring (pca_frozen unified)"
        pca, rim_debug = compute_opening_ring_pca_debug(env, vertices, env_id, frame=frame)
        rim_points = rim_debug.rim_points
        qc = rim_debug.quadrant_counts
        missing = [name for name in ("north", "south", "east", "west") if qc.get(name, 0) == 0]
        if missing:
            pca.warnings.append(f"opening_ring quadrants with zero samples: {', '.join(missing)}")
    else:
        cfg = extract_config or OpeningRimExtractConfig()
        origin = env.scene.env_origins[env_id].detach().cpu().tolist()
        cfg.env_origin = tuple(float(x) for x in origin)
        if hasattr(env, "cfg"):
            task_cfg = env.cfg
            cfg.opening_end_mode = str(getattr(task_cfg, "mouth_opening_end_mode", cfg.opening_end_mode))
            hint = getattr(task_cfg, "mouth_opening_toward_env_point", None)
            if hint is not None:
                cfg.opening_toward_env_point = tuple(float(x) for x in hint)
            cfg.end_slice_ratio = float(getattr(task_cfg, "mouth_end_slice_ratio", cfg.end_slice_ratio))
            cfg.entry_plane_ratio = float(getattr(task_cfg, "mouth_entry_plane_ratio", cfg.entry_plane_ratio))
            axis_hint = getattr(task_cfg, "mouth_axis_hint", None)
            if axis_hint is not None:
                cfg.axis_hint = tuple(float(x) for x in axis_hint)

        rim_debug = extract_opening_rim_points_geometric(vertices, cfg, return_debug=True)
        assert isinstance(rim_debug, RimExtractResult)
        rim_points = rim_debug.rim_points
        pca = compute_pca(rim_points)
        pca.axis_interpretation = interpret_pca_axes(pca.pc1, pca.pc2, pca.pc3, frame_label=frame)
        pca.nsew_candidates = _nsew_candidates_from_pca(rim_points, pca)
        u_ax, v_ax = _opening_plane_axes_from_pca(pca)
        loop_c = rim_debug.in_plane_center if rim_debug.in_plane_center is not None else pca.center
        semi_u, semi_v = _ellipse_semi_axes(loop_c, u_ax, v_ax, rim_points)
        rim_debug.rim_loop_viz = _synthesize_ellipse_rim(
            loop_c, u_ax, v_ax, semi_u, semi_v, n_samples=rim_debug.angular_bins_total or 72
        )
        if rim_debug.angular_bins_total > 0:
            frac = rim_debug.angular_bins_filled / rim_debug.angular_bins_total
            qc = rim_debug.quadrant_counts
            missing = [name for name in ("north", "south", "east", "west") if qc.get(name, 0) == 0]
            if frac < 0.55:
                pca.warnings.append(
                    f"angular coverage {rim_debug.angular_bins_filled}/{rim_debug.angular_bins_total} "
                    f"({frac:.0%}): hex mesh has sparse azimuth samples; "
                    "orange=solid measured loop, faint orange=ellipse reference."
                )
            if missing:
                pca.warnings.append(
                    f"rim quadrants with zero samples: {', '.join(missing)} "
                    "(mesh may lack vertices on that arc)."
                )
    pca.rim_extract = rim_debug

    text = pca.summary()
    text += f"\nrim_source: {rim_source}"
    text += f"\nrim_points: {rim_points.shape[0]} / {vertices.shape[0]} vertices"

    production_nsew: dict[str, torch.Tensor] | None = None
    compare = compare_debug_pca_with_pca_frozen(env, pca, env_id=env_id)
    if compare is not None:
        cmp_text = "\n".join(compare.lines)
        text += "\n" + cmp_text
        warn_thresh = 1.0 if use_ring else 15.0
        if compare.max_nsew_delta_mm > warn_thresh:
            pca.warnings.append(
                f"pca_frozen vs debug NSEW max Δ={compare.max_nsew_delta_mm:.2f} mm "
                f"(threshold {warn_thresh:.0f} mm)."
            )
        not_in_rim = [n for n, ok in compare.prod_index_in_debug_rim.items() if not ok]
        if not_in_rim:
            pca.warnings.append(
                f"production NSEW indices not in debug rim_points: {', '.join(not_in_rim)}"
            )
        if hasattr(env, "get_pca_frozen_nsew_snapshot"):
            snap = env.get_pca_frozen_nsew_snapshot(env_id)
            if snap is not None:
                production_nsew = snap["positions_env_local"]
                if use_ring and compare.max_nsew_delta_mm < 0.05:
                    production_nsew = None

    logger.info(text)
    print(text)

    if draw and cache_on_env:
        iface = _acquire_debug_draw_interface()
        env._opening_pca_draw_iface = iface
        env._opening_pca_viz = OpeningPCAVisualization(
            env_id=env_id,
            pca=pca,
            rim_points=rim_points,
            arrow_scale=arrow_scale,
            frame=frame,
            rim_extract=rim_debug,
            rim_loop_viz=rim_debug.rim_loop_viz,
            production_nsew=production_nsew,
        )
        if iface is not None:
            refresh_opening_pca_debug_draw(env)

    return pca


def explain_pca_interpretation() -> str:
    """Human-readable notes on PCA semantics for bracelet openings."""
    return """
=== PCA interpretation (bracelet opening rim) ===

1. "Direction of largest spread" (pc1 / λ1)
   - Among 3 orthogonal directions, the one where rim vertices vary the most.
   - On a cuff opening, this is usually the **wide diameter** (thumb–pinky), not "North".

2. Eigenvalue magnitude
   - λ1 >> λ2 >> λ3: good planar elliptical ring (stable axes).
   - λ1 ≈ λ2: nearly circular opening → pc1/pc2 rotate easily (unstable N/S/E/W).
   - λ3 large: ring is thick or extraction included non-rim points.

3. pc1 / pc2 / pc3 on the opening
   - pc1: widest in-plane axis (often E–W candidates).
   - pc2: narrower in-plane axis (often N–S candidates).
   - pc3: smallest variance ≈ opening-plane normal.

4. Rim extraction debug layers
   - Gray dots: all mesh vertices.
   - Gray grid + cross: opening plane candidate.
   - Magenta: plane-distance band (expanded end cap).
   - Yellow-green: radial outer annulus (before angular bins).
   - Cyan + solid orange loop: final rim_points.
   - Faint orange loop: fitted ellipse reference.

5. Debug draw checklist
   - Magenta cloud should wrap the opening face (not two vertical strips).
   - Cyan rim should span E/W; N/S depend on hex mesh density.
   - Solid orange loop follows cyan; compare to faint ellipse for coverage.
"""
