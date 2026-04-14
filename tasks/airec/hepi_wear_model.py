# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: BSD-3-Clause
"""HEPi graph builder + encoder for Wear glove (uses GeometryRL ``HEPi`` unchanged).

Aligned with cloth-hanging stack in GeometryRL, e.g.
``configs/cloth_hanging_multi_empn_trpl_cfg.yaml`` (Ponita GCN policy path) and
``configs/algorithm/pyg_agent/model/hepi.yaml`` (``latent_dim``, ``num_messages: 2``, ``concat_global: False``,
``ponita_dim: 3``, FiberBundleConv layout).

Requires: ``torch_geometric``, sibling repo ``geometry_rl`` on ``PYTHONPATH`` (see ``_CODE_ROOT``).
Install PyG: ``pip install torch-geometric torch-scatter torch-cluster`` (match your torch/cuda).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

_GRL_ROOT = Path(__file__).resolve().parents[2]
_CODE_ROOT = _GRL_ROOT.parent
if str(_CODE_ROOT / "geometry_rl") not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT / "geometry_rl"))

from geometry_rl.modules.pyg_models.hepi import HEPi  # noqa: E402
from geometry_rl.modules.pyg_models.ponita.conv import FiberBundleConv  # noqa: E402

try:
    import torch_scatter  # noqa: F401 — required by PyG knn / GeometryRL Ponita
    from torch_geometric.data import Batch, HeteroData
    from torch_geometric.nn import knn_graph
    from torch_geometric.nn.pool import global_mean_pool
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Wear HEPi needs torch-geometric plus torch-scatter and torch-cluster wheels matching your "
        "PyTorch/CUDA build, e.g.\n"
        "  pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.10.0+cu128.html\n"
        "(adjust torch/cu version to match `python -c \"import torch; print(torch.__version__)\"`)."
    ) from e

# PyG / HEPi use string node and edge type keys (same as ``rigid_tasks_data``).
NT_OBJ = "object_geometry"
NT_ACT = "grippers"
EDGE_INTERNAL = (NT_OBJ, "internal", NT_OBJ)
EDGE_AGENT = (NT_ACT, "agent", NT_ACT)
EDGE_TASK = (NT_OBJ, "task", NT_ACT)


def _make_fiber_convs(device: torch.device, latent_dim: int = 64):
    """Three edge levels × two message steps; pattern from ``hepi.yaml``."""

    def conv() -> FiberBundleConv:
        return FiberBundleConv(
            in_channels=latent_dim,
            out_channels=latent_dim,
            attr_dim=latent_dim,
            groups=latent_dim,
            separable=True,
            widening_factor=4,
        ).to(device)

    return [
        [conv(), None],
        [None, conv()],
        [None, conv()],
    ]


def build_wear_hepi(
    device: torch.device,
    *,
    latent_dim: int = 64,
    num_ori: int = 16,
    input_dim_node: int = 5,
) -> HEPi:
    """Instantiate :class:`HEPi` (same processor layout as ``configs/algorithm/pyg_agent/model/hepi.yaml``)."""
    mp = _make_fiber_convs(device, latent_dim=latent_dim)
    edge_type_mapping = [EDGE_INTERNAL, EDGE_AGENT, EDGE_TASK]
    edge_level_mapping = ["internal", "agent", "task"]

    # ``node_type_mapping`` is only stored on HEPi; edge matching uses string tuples above.
    class _NT:
        OBJECT = NT_OBJ
        ACTUATOR = NT_ACT

    model = HEPi(
        input_dim_node=input_dim_node,
        input_dim_edge=latent_dim,
        hidden_dim=latent_dim,
        latent_dim=latent_dim,
        output_dim=1,
        output_dim_vec=1,
        node_encoder_layers=2,
        edge_encoder_layers=2,
        node_decoder_layers=2,
        node_type_mapping=_NT,
        edge_type_mapping=edge_type_mapping,
        edge_level_mapping=edge_level_mapping,
        message_passing=mp,
        num_messages=2,
        concat_global=False,
        shared_processor=False,
        device=device,
        num_ori=num_ori,
        basis_dim=None,
        degree=2,
        ponita_dim=3,
        only_upper_hemisphere=False,
    )
    return model


def _one_env_hetero_data(
    glove_pos: torch.Tensor,
    ee_pos: torch.Tensor,
    knn_k: int,
    device: torch.device,
) -> HeteroData:
    """Single-env graph: glove KNN + gripper all-to-all + bipartite glove→EE."""
    v = glove_pos.shape[0]
    data = HeteroData()
    data[NT_OBJ].pos = glove_pos
    data[NT_ACT].pos = ee_pos

    if v > 1:
        k = max(1, min(knn_k, v - 1))
        ei = knn_graph(glove_pos, k=k, loop=False)
        if ei.shape[1] == 0:
            ei = torch.tensor([[0], [0]], dtype=torch.long, device=device)
    else:
        ei = torch.tensor([[0], [0]], dtype=torch.long, device=device)
    data[EDGE_INTERNAL].edge_index = ei

    if ee_pos.shape[0] > 1:
        data[EDGE_AGENT].edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    else:
        data[EDGE_AGENT].edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

    n_ee = ee_pos.shape[0]
    src = torch.arange(v, device=device).repeat_interleave(n_ee)
    dst = torch.arange(n_ee, device=device).repeat(v)
    data[EDGE_TASK].edge_index = torch.stack([src, dst], dim=0)

    data.output_mask_key = NT_OBJ
    return data


def build_u_dict(
    glove_pos: torch.Tensor,
    glove_vel: torch.Tensor,
    ee_pos: torch.Tensor,
    ee_vel: torch.Tensor,
    target_mean: torch.Tensor,
    device: torch.device,
) -> Tuple[dict, dict]:
    """Scalar (N,1) + vector (N,4,3) per type for ``HEPi.one_step``."""
    corr_g = target_mean.unsqueeze(0).expand_as(glove_pos) - glove_pos
    scalar_o = torch.ones(glove_pos.shape[0], 1, device=device)
    vec_o = torch.stack([glove_pos, corr_g, glove_vel, torch.zeros_like(glove_pos)], dim=1)

    corr_a = target_mean.unsqueeze(0).expand_as(ee_pos) - ee_pos
    scalar_a = torch.ones(ee_pos.shape[0], 1, device=device)
    vec_a = torch.stack([ee_pos, corr_a, ee_vel, torch.zeros_like(ee_pos)], dim=1)

    scalar_dict = {NT_OBJ: scalar_o, NT_ACT: scalar_a}
    vector_dict = {NT_OBJ: vec_o, NT_ACT: vec_a}
    return scalar_dict, vector_dict


class WearHepiEncoder(nn.Module):
    """Batch of envs → ``Batch`` → ``HEPi`` → global mean pool on glove nodes → ``(N, latent_dim)``.

    PPO minibatches can be tens of samples; one fused PyG ``Batch`` over all of them multiplies peak
    VRAM by batch size (nodes ≈ ``N × V``). ``forward_chunk`` caps graphs per ``one_step`` call.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        latent_dim: int = 64,
        knn_k: int = 8,
        input_dim_node: int = 5,
        forward_chunk: int = 2,
        num_ori: int = 16,
    ) -> None:
        super().__init__()
        self.knn_k = knn_k
        self.latent_dim = latent_dim
        self.forward_chunk = max(1, int(forward_chunk))
        self.hepi = build_wear_hepi(
            device,
            latent_dim=latent_dim,
            input_dim_node=input_dim_node,
            num_ori=max(4, int(num_ori)),
        )
        self.output_dim = latent_dim

    def forward_from_tensors(
        self,
        glove_pos: torch.Tensor,
        glove_vel: torch.Tensor,
        ee_pos: torch.Tensor,
        ee_vel: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Shapes: ``(N,V,3)``, ``(N,V,3)``, ``(N,2,3)``, ``(N,2,3)``, ``(N,T,3)``."""
        n = glove_pos.shape[0]
        dev = glove_pos.device
        zs: list[torch.Tensor] = []
        chunk = self.forward_chunk
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            graphs: list[HeteroData] = []
            uds: list[Tuple[dict, dict]] = []
            for i in range(start, end):
                tm = target_pos[i].mean(dim=0)
                g = _one_env_hetero_data(glove_pos[i], ee_pos[i], self.knn_k, dev)
                graphs.append(g)
                uds.append(build_u_dict(glove_pos[i], glove_vel[i], ee_pos[i], ee_vel[i], tm, dev))

            big = Batch.from_data_list(graphs)
            big.output_mask_key = NT_OBJ

            s_o = torch.cat([u[0][NT_OBJ] for u in uds], dim=0)
            s_a = torch.cat([u[0][NT_ACT] for u in uds], dim=0)
            v_o = torch.cat([u[1][NT_OBJ] for u in uds], dim=0)
            v_a = torch.cat([u[1][NT_ACT] for u in uds], dim=0)
            u_dict = ({NT_OBJ: s_o, NT_ACT: s_a}, {NT_OBJ: v_o, NT_ACT: v_a})

            _, latent_nodes = self.hepi.one_step(big, u_dict)
            batch_idx = big[NT_OBJ].batch
            zs.append(global_mean_pool(latent_nodes, batch_idx))

        return torch.cat(zs, dim=0)
