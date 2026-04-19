"""
Convex decomposition for AIREC meshes (.dae → coacd → .stl).

``.dae`` loading in trimesh needs **pycollada**::

    pip install pycollada

Also: ``pip install coacd trimesh`` (and any trimesh soft-deps you use).

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
DEFAULT_INPUT_ROOT = Path("/home/tamon/code/AIREC2_discription/torobo_resources/meshes")
DEFAULT_OUTPUT_ROOT = Path("/home/tamon/code/AIREC2_discription/torobo_resources/meshes_decomposed")

# def _load_trimesh(path: Path) -> trimesh.Trimesh:
#     loaded = trimesh.load(str(path), force="mesh")
#     if isinstance(loaded, trimesh.Scene):
#         geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
#         if not geoms:
#             raise ValueError(f"No Trimesh geometry in Scene: {path}")
#         loaded = trimesh.util.concatenate(geoms)
#     if not isinstance(loaded, trimesh.Trimesh):
#         raise TypeError(f"Expected Trimesh, got {type(loaded)} for {path}")
#     return loaded
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

# def clean_mesh_for_coacd(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
#     """Reduce CoACD crashes from dirty CAD / COLLADA imports."""
#     m = mesh.copy()
#     m.update_faces(m.unique_faces())
#     m.update_faces(m.nondegenerate_faces())
#     m.remove_unreferenced_vertices()
#     m.remove_unreferenced_vertices()
#     m.merge_vertices()
#     m.update_faces(m.unique_faces())
#     m.update_faces(m.nondegenerate_faces())
#     m.remove_unreferenced_vertices()
#     m.remove_unreferenced_vertices()
#     # COLLADA imports occasionally carry NaNs; CoACD can crash on them.
#     if not np.isfinite(m.vertices).all():
#         m.vertices = np.nan_to_num(np.asarray(m.vertices), nan=0.0, posinf=0.0, neginf=0.0)
#         m.merge_vertices()
#         m.remove_degenerate_faces()
#         m.remove_unreferenced_vertices()
#     if len(m.faces) == 0 or len(m.vertices) < 4:
#         raise ValueError("Mesh empty after cleaning")
#     return m
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

def run_coacd_on_mesh(mesh: trimesh.Trimesh, preprocess_mode: str) -> list:
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    # kwargs vary slightly by coacd version; ignore unknown for robustness
    kwargs = dict(
        threshold=0.05,
        preprocess_mode=preprocess_mode,
        preprocess_resolution=50,
    )
    try:
        return coacd.run_coacd(coacd_mesh, **kwargs)
    except TypeError:
        return coacd.run_coacd(coacd_mesh)

# def process_one_dae(dae_path: Path, output_path: Path) -> None:
#     raw = _load_trimesh(dae_path)
#     print(f"[debug] raw mesh: {dae_path}")
#     print(f"[debug] vertices={len(raw.vertices)}, faces={len(raw.faces)}")
#     print(f"[debug] is_empty={raw.is_empty}, is_watertight={raw.is_watertight}")

#     mesh = clean_mesh_for_coacd(raw)

#     print(f"[debug] cleaned mesh: {dae_path}")
#     print(f"[debug] vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")
    
#     last_err: Exception | None = None
#     parts = None
#     for pm in ("auto", "off", "on"):
#         try:
#             parts = run_coacd_on_mesh(mesh, preprocess_mode=pm)
#             break
#         except Exception as e:
#             last_err = e
#     if parts is None:
#         raise RuntimeError(f"CoACD failed for all preprocess modes; last error: {last_err}") from last_err

#     composite_mesh = trimesh.Scene()
#     for v, f in parts:
#         composite_mesh.add_geometry(trimesh.Trimesh(vertices=v, faces=f))
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     composite_mesh.export(str(output_path))

def process_one_dae(dae_path: Path, output_path: Path) -> None:
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
            parts = run_coacd_on_mesh(mesh, preprocess_mode=pm)
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


def batch_main(input_root: Path, output_root: Path) -> None:
    dae_files = sorted(input_root.rglob("*.dae"))
    print(f"Found {len(dae_files)} DAE files. Starting decomposition (one subprocess per file)...")

    for dae_path in dae_files:
        relative_path = dae_path.relative_to(input_root)
        output_path = output_root / relative_path.with_suffix(".stl")
        print(f"Processing: {relative_path}")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(dae_path), str(output_path)]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"[batch] SKIP (worker exit {proc.returncode}): {dae_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch CoACD convex decomposition for .dae under a folder.")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("dae", nargs="?", type=Path, help="(with --worker) input .dae")
    ap.add_argument("out", nargs="?", type=Path, help="(with --worker) output .stl")
    args = ap.parse_args()

    if args.worker:
        if args.dae is None or args.out is None:
            ap.error("--worker requires dae and out positional arguments")
        try:
            process_one_dae(args.dae, args.out)
        except Exception as e:
            print(f"[worker] FAILED {args.dae}: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"[worker] OK {args.dae} -> {args.out}")
        sys.exit(0)

    batch_main(args.input_root, args.output_root)
    print("Done!")
