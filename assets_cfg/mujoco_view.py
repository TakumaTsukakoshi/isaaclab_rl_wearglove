# # ファイル名は mujoco.py にしないでください（名前衝突のため）
# import time
# import numpy as np
# import mujoco
# import mujoco.viewer

# # 重要: URDFは後述の通りアクチュエータが無い場合が多い
# model = mujoco.MjModel.from_xml_path("nextage.xml")
# data = mujoco.MjData(model)

# print(f"nq={model.nq}, nv={model.nv}, nu(=actuators)={model.nu}")

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     # 実時間に近い速度で回す
#     while viewer.is_running():
#         # 例: アクチュエータが1つ以上あれば、0番に弱い正弦制御を入れて様子を見る
#         if model.nu > 0:
#             data.ctrl[:] = 0.0
#             data.ctrl[0] = 0.2 * np.sin(2 * np.pi * 0.5 * data.time)

#         mujoco.mj_step(model, data)
#         viewer.sync()
#         time.sleep(model.opt.timestep)  # ペース調整
import time
import mujoco
import mujoco.viewer
from dm_control import mjcf

import numpy as np

path="/home/takuma/code/isaaclab_rl_wearglove/assets/nextage_description/urdf/meshes/nextage_isaaclab_edit_mujoco.urdf"
model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(model)

# mjcf_model = mjcf.from_path(path)
# xml_str = mjcf_model.to_xml_string()

# with open("converted_model.xml", "w") as f:
#     f.write(xml_str)
# this is the “interactive” viewer in 3.x
mujoco.viewer.launch(model, data)