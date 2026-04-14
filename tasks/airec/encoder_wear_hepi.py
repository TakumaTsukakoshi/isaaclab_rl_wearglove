# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Multimodal encoder: prop/gt/tactile MLP early fusion + HEPi global latent (Wear glove)."""

from __future__ import annotations

import torch
import torch.nn as nn

from multimodal_rl.models.mlp import MLP

from tasks.airec.hepi_wear_model import WearHepiEncoder


class WearHepiFusionEncoder(nn.Module):
    """Concatenate early-fused state vectors with HEPi pooled embedding, then MLP."""

    def __init__(self, observation_space, action_space, env_cfg, config_dict, device) -> None:
        super().__init__()
        self.device = device
        self.observation_space = observation_space
        enc_cfg = config_dict["encoder"]
        hepi_cfg = enc_cfg.get("hepi", {})
        self.hepi_enc = WearHepiEncoder(
            device,
            latent_dim=int(hepi_cfg.get("latent_dim", 64)),
            knn_k=int(hepi_cfg.get("knn_k", 8)),
            input_dim_node=int(hepi_cfg.get("input_dim_node", 5)),
            forward_chunk=int(hepi_cfg.get("forward_chunk", 1)),
            num_ori=int(hepi_cfg.get("num_ori", 16)),
        )

        state_dim = 0
        for k in sorted(observation_space.keys()):
            if k in ("prop", "gt", "tactile"):
                state_dim += observation_space[k].shape[0]

        hiddens = enc_cfg["hiddens"]
        acts = enc_cfg["activations"]
        layernorm = enc_cfg["layernorm"]
        fusion_in = state_dim + self.hepi_enc.output_dim
        self.net = MLP(fusion_in, hiddens, acts, layernorm=layernorm).to(device)
        self.num_outputs = hiddens[-1]
        self.state_preprocessor = None
        if enc_cfg.get("state_preprocessor") is not None:
            from multimodal_rl.models.running_standard_scaler import RunningStandardScalerDict

            self.state_preprocessor = RunningStandardScalerDict(size=observation_space, device=device)

        # Move HEPi + fusion head together: ``hepi_enc`` was only constructed on ``device`` by argument but
        # submodules default to CPU unless the root module is ``.to(device)`` (same pattern as :class:`Encoder`).
        self.to(device)

    def forward(self, obs_dict, detach=False, train=False):
        if "policy" in obs_dict:
            obs_dict = obs_dict["policy"]
        if detach:
            obs_dict = {k: v.detach() for k, v in obs_dict.items()}
        if self.state_preprocessor is not None:
            obs_dict = self.state_preprocessor(obs_dict, train)

        g = obs_dict["hepi_glove_pos"].view(obs_dict["hepi_glove_pos"].shape[0], -1, 3)
        gv = obs_dict["hepi_glove_vel"].view_as(g)
        ee = obs_dict["hepi_ee_pos"].view(obs_dict["hepi_ee_pos"].shape[0], 2, 3)
        ev = obs_dict["hepi_ee_vel"].view(obs_dict["hepi_ee_vel"].shape[0], 2, 3)
        tg = obs_dict["hepi_target_pos"].view(obs_dict["hepi_target_pos"].shape[0], -1, 3)
        z_h = self.hepi_enc.forward_from_tensors(g, gv, ee, ev, tg)
        z_h = torch.nan_to_num(z_h, nan=0.0, posinf=0.0, neginf=0.0)

        rest = []
        for k in sorted(obs_dict.keys()):
            if k in ("prop", "gt", "tactile"):
                rest.append(obs_dict[k])
        z_r = torch.cat(rest, dim=-1) if rest else torch.empty(z_h.shape[0], 0, device=z_h.device)
        z_r = torch.nan_to_num(z_r, nan=0.0, posinf=0.0, neginf=0.0)
        fused = torch.nan_to_num(torch.cat([z_r, z_h], dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        return self.net(fused)
