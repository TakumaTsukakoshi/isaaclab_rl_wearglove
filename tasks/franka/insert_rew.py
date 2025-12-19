import torch
from typing import Optional, Dict, Tuple

class InsertReward:
    """
    Batched 'insert by 3 cm and hold' detector + controller.

    - State machine: ALIGN(0) -> ENTER(1) -> INSERT(2) -> HOLD(3=success)
    - Geometry in edge frame {E}; commands returned in the same source frame {S} you pass in.
      (e.g., set {S} = robot base if your FrameTransformers share the same source.)
    - All inputs are batched; use `idx` to update internal buffers for a subset of envs.
    """

    def __init__(self,
                 num_envs: int,
                 device: torch.device = torch.device("cpu"),
                 # thresholds
                 d_enter: float = 0.010,   # [m] mouth-plane crossing
                 d_goal:  float = 0.030,   # [m] target depth (3 cm)
                 eps_d:   float = 0.002,   # [m] tolerance at goal
                 r_align: float = 0.010,   # [m] lateral tol for align/enter
                 r_stay:  float = 0.013,   # [m] lateral tol for hold
                 c_align: float = 0.90,    # cos threshold for align/enter
                 c_stay:  float = 0.90,    # cos threshold for hold
                 dwell_min: float = 0.30,  # [s] success debounce
                 # control gains & limits
                 k_d: float = 1.0, k_r: float = 3.0, k_c: float = 2.0,
                 v_max: float = 0.10, w_max: float = 0.5,
                 # inward direction assumption if no center hint is provided
                 inward_assume: str = "-x",   # '-x': +x_E is outward (common), inward is -x
                 # gripper tip model (in gripper frame {G})
                 p_tip_in_G: Tuple[float, float, float] = (0.0, 0.0, 0.0),  # set e.g. (0.046,0,0)
                 t_axis_in_G: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # +x_G is forward
                 ):
        self.N = num_envs
        self.device = device

        # thresholds
        self.d_enter = d_enter
        self.d_goal  = d_goal
        self.eps_d   = eps_d
        self.r_align = r_align
        self.r_stay  = r_stay
        self.c_align = c_align
        self.c_stay  = c_stay
        self.dwell_min = dwell_min

        # control
        self.k_d, self.k_r, self.k_c = k_d, k_r, k_c
        self.v_max, self.w_max = v_max, w_max

        # states/buffers (full env)
        self.state   = torch.full((self.N,), 0, dtype=torch.int64, device=device)   # 0..3
        self.dwell   = torch.zeros(self.N, dtype=torch.float32, device=device)      # [s]
        self.success = torch.zeros(self.N, dtype=torch.bool, device=device)

        # inward sign (+1 means +x_E is inward, -1 means -x_E is inward)
        self.sgn = torch.full((self.N,), -1.0 if inward_assume == "-x" else 1.0,
                              dtype=torch.float32, device=device)
        self.sgn_initialized = torch.zeros(self.N, dtype=torch.bool, device=device)

        # tip model in {G}
        self._p_tip_in_G = torch.tensor(p_tip_in_G, device=device, dtype=torch.float32)  # (3,)
        t_axis = torch.tensor(t_axis_in_G, device=device, dtype=torch.float32)
        self._t_axis_in_G = t_axis / (t_axis.norm() + 1e-9)                              # (3,)

    # -------------------------- math utils --------------------------

    @staticmethod
    def _quat_wxyz_to_R(q: torch.Tensor) -> torch.Tensor:
        """
        q: (B,4) in [w,x,y,z]
        return R: (B,3,3) rotation matrices (source <- frame)
        """
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        two = 2.0
        xx, yy, zz = x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z

        R = torch.stack([
            torch.stack([1 - two*(yy + zz), two*(xy - wz),   two*(xz + wy)], dim=-1),
            torch.stack([two*(xy + wz),     1 - two*(xx + zz),   two*(yz - wx)], dim=-1),
            torch.stack([two*(xz - wy),     two*(yz + wx),       1 - two*(xx + yy)], dim=-1)
        ], dim=1)
        return R  # (B,3,3)

    @staticmethod
    def _sat_vec(v: torch.Tensor, vmax: float) -> torch.Tensor:
        """
        v: (B,3), saturate by vector 2-norm per row
        """
        n = torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-9)
        scale = torch.clamp(vmax / n, max=1.0)
        return v * scale

    # -------------------------- core step --------------------------

    @torch.no_grad()
    def step(self,
             pos_edge_s: torch.Tensor,   # (B,3)  edge/mouth pose in source {S}
             quat_edge_s: torch.Tensor,  # (B,4)  (wxyz)
             pos_grip_s: torch.Tensor,   # (B,3)
             quat_grip_s: torch.Tensor,  # (B,4)
             dt: torch.Tensor,           # (B,)   measured time step [s]
             center_hint_s: Optional[torch.Tensor] = None,  # (B,3) known-inside point in {S}
             idx: Optional[torch.Tensor] = None             # (B,) env indices into internal buffers
             ) -> Dict[str, torch.Tensor]:
        """
        Returns:
          v_s, w_s : (B,3) linear & angular velocity in source {S}
          d, r, c  : (B,) depth [m], lateral [m], axis alignment cos
          state    : (B,) current discrete state (0..3)
          success  : (B,) bool
          dwell    : (B,) accumulated success dwell time [s]
        """
        # --- batch size & index mapping ---
        B = pos_edge_s.shape[0]
        if idx is None:
            if B != self.N:
                raise ValueError(f"Batch {B} != N {self.N}. Pass idx=(B,) to map into buffers.")
            idx = torch.arange(self.N, device=self.device)
        else:
            if idx.shape[0] != B:
                raise ValueError(f"idx length {idx.shape[0]} must match batch {B}.")

        # --- rotations & transforms in {S} ---
        R_SE = self._quat_wxyz_to_R(quat_edge_s)      # source <- edge
        R_ES = R_SE.transpose(1, 2)                   # edge   <- source
        R_SG = self._quat_wxyz_to_R(quat_grip_s)      # source <- grip

        # tip position & forward axis in {S}  (use CAD tip offset if provided)
        p_tip_s = pos_grip_s + torch.bmm(R_SG, self._p_tip_in_G.view(1,3).expand(B,-1).unsqueeze(-1)).squeeze(-1)
        t_axis_s = torch.bmm(R_SG, self._t_axis_in_G.view(1,3).expand(B,-1).unsqueeze(-1)).squeeze(-1)  # (B,3)

        # into edge frame {E}
        v_se   = (p_tip_s - pos_edge_s).unsqueeze(-1)           # (B,3,1)
        p_tip_e = torch.bmm(R_ES, v_se).squeeze(-1)             # (B,3)
        t_axis_e = torch.bmm(R_ES, t_axis_s.unsqueeze(-1)).squeeze(-1)  # (B,3)
        t_axis_e = t_axis_e / (torch.linalg.norm(t_axis_e, dim=1, keepdim=True) + 1e-9)

        # auto-initialize inward sign if a center hint is provided
        if center_hint_s is not None:
            n_raw_s = R_SE[:, :, 0]  # +x_E in {S}
            to_center = center_hint_s - pos_edge_s
            s = torch.sign((to_center * n_raw_s).sum(dim=1)).clamp(min=-1.0, max=1.0)
            unset = ~self.sgn_initialized[idx]
            # write only where not set
            new_sgn = torch.where(unset, s.to(self.sgn.dtype), self.sgn[idx])
            self.sgn[idx] = new_sgn
            self.sgn_initialized[idx] = self.sgn_initialized[idx] | unset

        # metrics (depth d, lateral r, align cos c), apply inward sign consistently
        sgn_b = self.sgn[idx]  # (B,)
        d = sgn_b * p_tip_e[:, 0]
        r = torch.linalg.vector_norm(p_tip_e[:, 1:3], dim=1)
        c = (sgn_b * t_axis_e[:, 0]).clamp(-1.0, 1.0)
        import ipdb; ipdb.set_trace()

        # inward +x_E for rotation target
        xE = torch.tensor([1.0, 0.0, 0.0], device=self.device).view(1,3).expand(B,3)
        xE = sgn_b.view(B,1) * xE

        # ---------------- state machine ----------------
        ALIGN, ENTER, INSERT, HOLD = 0, 1, 2, 3
        v_e = torch.zeros((B,3), device=self.device)
        w_e = torch.zeros((B,3), device=self.device)

        # ALIGN → ENTER
        is_align = (self.state[idx] == ALIGN)
        ready_align = (c >= self.c_align) & (r <= self.r_align)
        self.state[idx] = torch.where(is_align & ready_align,
                                      torch.full_like(self.state[idx], ENTER),
                                      self.state[idx])

        # ENTER → INSERT
        is_enter = (self.state[idx] == ENTER)
        enter_ok = (d >= self.d_enter) & (r <= self.r_align) & (c >= self.c_align)
        self.state[idx] = torch.where(is_enter & enter_ok,
                                      torch.full_like(self.state[idx], INSERT),
                                      self.state[idx])

        # INSERT: drive to d_goal, damp lateral; HOLD on dwell
        is_insert = (self.state[idx] == INSERT)
        at_goal = (torch.abs(d - self.d_goal) <= self.eps_d) & (r <= self.r_stay) & (c >= self.c_stay)

        self.dwell[idx] = torch.where(
            is_insert & at_goal,
            torch.minimum(self.dwell[idx] + dt, torch.full_like(self.dwell[idx], self.dwell_min)),
            torch.where(is_insert, torch.zeros_like(self.dwell[idx]), self.dwell[idx])
        )
        become_hold = is_insert & (self.dwell[idx] >= self.dwell_min)
        self.state[idx] = torch.where(become_hold,
                                      torch.full_like(self.state[idx], HOLD),
                                      self.state[idx])

        self.success[idx] = (self.state[idx] == HOLD)
        success_b = self.success[idx]
        not_success = ~success_b

        # --- control laws in {E} ---
        # ALIGN: lateral damp
        ctrl_align = (self.state[idx] == ALIGN) & not_success
        v_e[ctrl_align, 0] = 0.0
        v_e[ctrl_align, 1] = -self.k_r * p_tip_e[ctrl_align, 1]
        v_e[ctrl_align, 2] = -self.k_r * p_tip_e[ctrl_align, 2]

        # ENTER: go to d_enter
        ctrl_enter = (self.state[idx] == ENTER) & not_success
        e_depth_enter = (self.d_enter - d).clamp_min(-0.05).clamp_max(0.05)
        v_e[ctrl_enter, 0] = self.k_d * e_depth_enter[ctrl_enter]
        v_e[ctrl_enter, 1] = -self.k_r * p_tip_e[ctrl_enter, 1]
        v_e[ctrl_enter, 2] = -self.k_r * p_tip_e[ctrl_enter, 2]

        # INSERT: go to d_goal
        ctrl_insert = (self.state[idx] == INSERT) & not_success
        e_depth_goal = (self.d_goal - d).clamp_min(-0.05).clamp_max(0.05)
        v_e[ctrl_insert, 0] = self.k_d * e_depth_goal[ctrl_insert]
        v_e[ctrl_insert, 1] = -self.k_r * p_tip_e[ctrl_insert, 1]
        v_e[ctrl_insert, 2] = -self.k_r * p_tip_e[ctrl_insert, 2]

        # rotation (ALIGN/ENTER/INSERT): omega_e = k_c * (t_axis_e x xE)
        need_rot = (ctrl_align | ctrl_enter | ctrl_insert)
        if need_rot.any():
            cross = torch.cross(t_axis_e, xE, dim=1)
            w_e[need_rot] = self.k_c * cross[need_rot]

        # {E} → {S}
        v_s = torch.bmm(R_SE, v_e.unsqueeze(-1)).squeeze(-1)
        w_s = torch.bmm(R_SE, w_e.unsqueeze(-1)).squeeze(-1)

        # saturation
        v_s = self._sat_vec(v_s, self.v_max)
        w_s = self._sat_vec(w_s, self.w_max)

        # stop if success
        v_s = torch.where(success_b.view(B,1), torch.zeros_like(v_s), v_s)
        w_s = torch.where(success_b.view(B,1), torch.zeros_like(w_s), w_s)

        return {
            "v_s": v_s, "w_s": w_s,
            "d": d, "r": r, "c": c,
            "state": self.state[idx],
            "success": success_b,
            "dwell": self.dwell[idx],
        }
