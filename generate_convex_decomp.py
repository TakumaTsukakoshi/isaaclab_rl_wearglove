"""
Convex decomposition for AIREC meshes (.dae → coacd → .stl).

``.dae`` loading in trimesh needs **pycollada**::

    pip install pycollada

Also: ``pip install coacd trimesh`` (and any trimesh soft-deps you use).

Default: **hand collision only** (``meshes/hand/``, URDF variants TRBD-LH/RH-B-001-6).
Torso/arm are left on original ``meshes/.../STL`` so link self-collision stays strict.

Example (airec2_finger hands only)::

    python generate_convex_decomp.py

    python generate_convex_decomp.py --all-hand-variants   # all hand/ subfolders
    python generate_convex_decomp.py --parts hand --hand-variants TRBD-LH-B-001-6,TRBD-RH-B-001-6

Output: ``torobo_resources/meshes_decomposed/hand/.../*.stl``

Point URDF **collision** (not visual) at decomposed STLs for hand links only.

Why you can see ``std::length_error`` / ``Aborted (core dumped)``
---------------------------------------------------------------
CoACD is mostly **C++**. If its manifold / preprocessing code hits bad
topology (non-manifold export, broken connectivity, degenerate faces) or an
internal edge case, it can **abort the whole process**. Python ``try/except``
does **not** catch that.

This script therefore:
- **Cleans** the mesh with trimesh before calling CoACD.
- Runs each file in a **subprocess** so one bad ``.dae`` does not kill the batch.
- Optionally retries with ``preprocess_mode="off"`` if the first attempt fails.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import coacd
import trimesh

try:
    import collada  # noqa: F401 — used by trimesh for COLLADA / .dae
except ImportError as e:
    raise SystemExit(
        "Cannot load .dae files: install pycollada in this environment, then re-run:\n"
        "  pip install pycollada\n"
    ) from e

# --- Default paths (override with CLI) ---
# …/code/isaaclab_rl_wearglove/generate_convex_decomp.py → parents[1] == …/code
_CODE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TOROBO = _CODE_ROOT / "AIREC2_description" / "torobo_resources"
DEFAULT_INPUT_ROOT = _DEFAULT_TOROBO / "meshes"
DEFAULT_OUTPUT_ROOT = _DEFAULT_TOROBO / "meshes_decomposed"

# ``airec2_finger_new_fixed.urdf`` collision meshes (left + right Shadow-style hands).
DEFAULT_HAND_VARIANTS = ("TRBD-LH-B-001-6", "TRBD-RH-B-001-6")

KNOWN_PARTS = ("hand", "arm", "torso", "head", "base", "wheel", "gripper")


def _load_trimesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="mesh")

    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        print(f"[debug] Scene geometries in {path}: {len(geoms)}")
        if not geoms:
            raise ValueError(f"No Trimesh geometry in Scene: {path}")
        loaded = trimesh.util.concatenate(geoms)

    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(loaded)} for {path}")

    print(
        f"[debug] loaded {path}: vertices={len(loaded.vertices)}, "
        f"faces={len(loaded.faces)}, is_empty={loaded.is_empty}"
    )
    return loaded


def clean_mesh_for_coacd(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()

    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    m.merge_vertices()

    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()

    if len(m.vertices) > 0 and not np.isfinite(m.vertices).all():
        m.vertices = np.nan_to_num(
            np.asarray(m.vertices), nan=0.0, posinf=0.0, neginf=0.0
        )
        m.merge_vertices()
        m.update_faces(m.nondegenerate_faces())
        m.remove_unreferenced_vertices()

    if len(m.faces) == 0 or len(m.vertices) < 4:
        raise ValueError("Mesh empty after cleaning")

    return m


def run_coacd_on_mesh(mesh: trimesh.Trimesh, preprocess_mode: str, threshold: float) -> list:
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    kwargs = dict(
        threshold=threshold,
        preprocess_mode=preprocess_mode,
        preprocess_resolution=50,
    )
    try:
        return coacd.run_coacd(coacd_mesh, **kwargs)
    except TypeError:
        return coacd.run_coacd(coacd_mesh)


def process_one_dae(dae_path: Path, output_path: Path, *, threshold: float) -> None:
    raw = _load_trimesh(dae_path)

    if raw.is_empty or len(raw.faces) == 0:
        print(f"[worker] SKIP empty raw mesh: {dae_path}")
        return

    try:
        mesh = clean_mesh_for_coacd(raw)
    except ValueError as e:
        if "Mesh empty after cleaning" in str(e):
            print(f"[worker] SKIP empty cleaned mesh: {dae_path}")
            return
        raise

    last_err = None
    parts = None
    for pm in ("auto", "off", "on"):
        try:
            parts = run_coacd_on_mesh(mesh, preprocess_mode=pm, threshold=threshold)
            break
        except Exception as e:
            last_err = e

    if parts is None:
        raise RuntimeError(
            f"CoACD failed for all preprocess modes; last error: {last_err}"
        ) from last_err

    composite_mesh = trimesh.Scene()
    for v, f in parts:
        composite_mesh.add_geometry(trimesh.Trimesh(vertices=v, faces=f))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite_mesh.export(str(output_path))


def _parse_csv_list(value: str | None) -> tuple[str, ...] | None:
    if value is None or not str(value).strip():
        return None
    return tuple(x.strip() for x in str(value).split(",") if x.strip())


def iter_dae_files(
    input_root: Path,
    *,
    parts: tuple[str, ...],
    hand_variants: tuple[str, ...] | None,
) -> list[Path]:
    """Collect ``.dae`` under ``input_root/<part>/``, optionally filtering hand variants."""
    out: list[Path] = []
    for part in parts:
        part_dir = input_root / part
        if not part_dir.is_dir():
            print(f"[batch] WARNING: missing part directory: {part_dir}")
            continue
        for dae_path in sorted(part_dir.rglob("*.dae")):
            if part == "hand" and hand_variants is not None:
                rel = dae_path.relative_to(part_dir)
                if not rel.parts or rel.parts[0] not in hand_variants:
                    continue
            out.append(dae_path)
    return out


def batch_main(
    input_root: Path,
    output_root: Path,
    *,
    parts: tuple[str, ...],
    hand_variants: tuple[str, ...] | None,
    threshold: float,
) -> None:
    dae_files = iter_dae_files(input_root, parts=parts, hand_variants=hand_variants)
    if not dae_files:
        print(
            f"[batch] No .dae files matched parts={parts!r} "
            f"hand_variants={hand_variants!r} under {input_root}"
        )
        return

    print(
        f"Found {len(dae_files)} DAE file(s) for parts={list(parts)} "
        f"(hand_variants={list(hand_variants) if hand_variants else 'all'}). "
        "Starting decomposition (one subprocess per file)..."
    )

    ok = 0
    skip = 0
    for dae_path in dae_files:
        relative_path = dae_path.relative_to(input_root)
        output_path = output_root / relative_path.with_suffix(".stl")
        print(f"Processing: {relative_path}")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(dae_path),
            str(output_path),
            "--threshold",
            str(threshold),
        ]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"[batch] SKIP (worker exit {proc.returncode}): {dae_path}")
            skip += 1
        else:
            ok += 1

    print(f"[batch] finished: ok={ok} skip={skip} total={len(dae_files)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "CoACD convex decomposition. Default: hand meshes only "
            f"({', '.join(DEFAULT_HAND_VARIANTS)}) for grasp collision."
        )
    )
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument(
        "--parts",
        type=str,
        default="hand",
        help=(
            "Comma-separated mesh part folders under input-root "
            f"(known: {', '.join(KNOWN_PARTS)}). Default: hand only."
        ),
    )
    ap.add_argument(
        "--hand-variants",
        type=str,
        default=",".join(DEFAULT_HAND_VARIANTS),
        help=(
            "When --parts includes hand: only these subfolders under meshes/hand/. "
            "Use 'all' for every hand variant."
        ),
    )
    ap.add_argument(
        "--all-hand-variants",
        action="store_true",
        help="Process every meshes/hand/* variant (overrides --hand-variants).",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="CoACD concavity threshold (smaller → more convex pieces, tighter fit).",
    )
    ap.add_argument("dae", nargs="?", type=Path, help="(with --worker) input .dae")
    ap.add_argument("out", nargs="?", type=Path, help="(with --worker) output .stl")
    args = ap.parse_args()

    if args.worker:
        if args.dae is None or args.out is None:
            ap.error("--worker requires dae and out positional arguments")
        try:
            process_one_dae(args.dae, args.out, threshold=float(args.threshold))
        except Exception as e:
            print(f"[worker] FAILED {args.dae}: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"[worker] OK {args.dae} -> {args.out}")
        sys.exit(0)

    parts = tuple(p.strip() for p in args.parts.split(",") if p.strip())
    unknown = [p for p in parts if p not in KNOWN_PARTS]
    if unknown:
        ap.error(f"Unknown --parts: {unknown}. Known: {list(KNOWN_PARTS)}")

    hand_variants: tuple[str, ...] | None
    if "hand" not in parts:
        hand_variants = None
    elif args.all_hand_variants or (args.hand_variants or "").strip().lower() == "all":
        hand_variants = None
    else:
        hand_variants = _parse_csv_list(args.hand_variants) or DEFAULT_HAND_VARIANTS

    print(f"[batch] input-root:  {args.input_root.resolve()}")
    print(f"[batch] output-root: {args.output_root.resolve()}")
    batch_main(
        args.input_root,
        args.output_root,
        parts=parts,
        hand_variants=hand_variants,
        threshold=float(args.threshold),
    )
    print("Done!")
