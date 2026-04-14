"""
Convex decomposition for AIREC meshes (.dae → coacd → .stl).

``.dae`` loading in trimesh needs **pycollada**::

    pip install pycollada

Also: ``pip install coacd trimesh`` (and any trimesh soft-deps you use).
"""

from pathlib import Path

import coacd
import trimesh

try:
    import collada  # noqa: F401 — used by trimesh for COLLADA / .dae
except ImportError as e:
    raise SystemExit(
        "Cannot load .dae files: install pycollada in this environment, then re-run:\n"
        "  pip install pycollada\n"
    ) from e

# --- Configuration ---
# Set the root of your existing meshes
input_root = Path("/home/tamon/code/isaaclab_rl_wearglove/assets/airec_finger/meshes")
# Set where the decomposed meshes should go
output_root = Path("/home/tamon/code/isaaclab_rl_wearglove/assets/airec_finger/visual_meshes_decomposed2")

def process_meshes():
    # 1. Find all .stl files recursively
    # rglob stands for recursive glob
    dae_files = list(input_root.rglob("*.dae"))
    
    print(f"Found {len(dae_files)} DAE files. Starting decomposition...")

    for dae_path in dae_files:
        # 2. Determine the relative path to maintain folder structure
        # e.g., 'arm/TRBD-LA-A-001-7/STL/link1.stl'
        relative_path = dae_path.relative_to(input_root)
        output_path = output_root / relative_path.with_suffix(".stl")

        print(f"Output path: {output_path}")
        
        # 3. Create the subdirectories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing: {relative_path}")
        
        try:
            # 4. Load and Process
            mesh = trimesh.load(str(dae_path), force="mesh")
            coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
            parts = coacd.run_coacd(coacd_mesh)
            
            # 5. Merge parts into a single trimesh object and export
            # CoACD returns a list of (vertices, faces) tuples
            composite_mesh = trimesh.Scene()
            for v, f in parts:
                composite_mesh.add_geometry(trimesh.Trimesh(vertices=v, faces=f))
            
            # Export as STL (trimesh will merge scene geometries into one file)
            composite_mesh.export(str(output_path))
            
        except Exception as e:
            print(f"Error processing {dae_path}: {e}")

if __name__ == "__main__":
    process_meshes()
    print("Done!")