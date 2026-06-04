# EE Stretch Debug (Deformable Bracelet evaluation only)

Investigate excessive lateral EE separation during deformable bracelet insertion.
Logging and clamp apply **only** when `evaluation_mode=True` (set automatically by `play.py` / `debug_rollout.py`).

## Phase 1 investigation summary

```text
Evaluation entry point:
  play.py main(); debug_rollout.py main()

Policy action application point:
  AIRECEnv._pre_physics_step → _update_actuated_joint_pos_cmd_from_actions

Controller target generation point:
  joint_pos_cmd[:, actuated_dof_indices] (EMA + clamp); applied in _apply_action via set_joint_position_target

Left EE position tensor:
  left_upper_ee_pos (from left_upper_ee_frame.data.target_pos_source)

Right EE position tensor:
  right_upper_ee_pos

Left arm joint_7:
  robot index: robot.joint_names.index("left_arm_joint_7")
  policy column: actuated_joint_names.index("left_arm_joint_7")  # 6

Right arm joint_7:
  robot index: right_arm_joint_7; policy column 13

Actual joint position / velocity:
  robot.data.joint_pos / joint_vel at robot DOF indices

Commanded target:
  joint_pos_cmd at robot DOF indices

Policy action (joint_7):
  actions[:, 6] left, actions[:, 13] right

Deformable Bracelet identification:
  Gym task AIREC_Reach_Deformable_Bracelet; ReachDeformableBraceletEnv; object_type="deformable"

Reward components:
  extras["log"] in ReachDeformableBraceletEnv._get_rewards

Success:
  task_success = wrist_center_euclidean_distance < bracelet_success_threshold (0.01 m)

CLI:
  ee_stretch_cli.py + common_utils.update_env_cfg

Seed / initial condition:
  --seed passed to set_seed(); env_cfg.seed; env.reset(hard=True)
```

## Commands

Replace `<CHECKPOINT>` and optional flags.

### Condition A — baseline logging only

```bash
python play.py --task AIREC_Reach_Deformable_Bracelet --checkpoint <CHECKPOINT> \
  --seed 1234 --num_envs 1 --headless \
  --debug-ee-stretch-log \
  --debug-ee-stretch-log-dir logs/ee_stretch_debug/baseline \
  --debug-ee-watch-distance 0.25 \
  --debug-target-object deformable_bracelet
```

Or via `debug_rollout.py`:

```bash
python debug_rollout.py --checkpoint <CHECKPOINT> --seed 1234 --num_envs 1 --headless \
  --episodes 10 \
  --debug-ee-stretch-log \
  --debug-ee-stretch-log-dir logs/ee_stretch_debug/baseline \
  --debug-ee-watch-distance 0.25 \
  --debug-target-object deformable_bracelet
```

### Condition B — EE distance clamp enabled

```bash
python play.py --task AIREC_Reach_Deformable_Bracelet --checkpoint <CHECKPOINT> \
  --seed 1234 --num_envs 1 --headless \
  --debug-ee-stretch-log \
  --debug-ee-stretch-log-dir logs/ee_stretch_debug/clamp \
  --debug-ee-watch-distance 0.25 \
  --debug-enable-ee-distance-clamp \
  --debug-ee-clamp-limit 0.30 \
  --debug-ee-clamp-activation-distance 0.295 \
  --debug-ee-clamp-mode remove_outward_relative_command \
  --debug-target-object deformable_bracelet
```

### Compare aggregate summaries

```python
from pathlib import Path
from tasks.airec.ee_stretch_debug import compare_aggregate_summaries

compare_aggregate_summaries(
    Path("logs/ee_stretch_debug/baseline/aggregate_summary.json"),
    Path("logs/ee_stretch_debug/clamp/aggregate_summary.json"),
    Path("logs/ee_stretch_debug/comparison.json"),
)
```

## Output paths (under `--debug-ee-stretch-log-dir`)

| File | Content |
|------|---------|
| `step_log.csv` / `step_log.npz` | Per-step time series |
| `episode_summaries.json` | Per-episode milestones |
| `aggregate_summary.json` | Run-level stats |
| `analysis.json` | Correlations + outward direction candidates |
| `plots/ep*_*.png` | Step plots + scatter |
| `INTERPRETATION.md` | Interpretation criteria |

## Clamp implementation note

Cartesian EE command targets are **not** exposed. Default mode `remove_outward_relative_command` removes outward separation using **arm-wide joint_pos_cmd delta sums** as a diagnostic proxy (not Jacobian-exact). Optional `joint7_fallback` blocks outward `left/right_arm_joint_7` increments when `--debug-left-joint7-outward-direction` / `--debug-right-joint7-outward-direction` are set.

## Verification

- **Clamp OFF**: omit `--debug-enable-ee-distance-clamp`; code path skips `apply_ee_distance_clamp` (identical to prior evaluation).
- **Training**: `train.py` does not set `evaluation_mode`; clamp and logging are disabled even if flags were passed.

## Interpretation criteria

See `INTERPRETATION.md` in each log directory after a run.
