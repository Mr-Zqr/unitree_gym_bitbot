import os
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
# from legged_gym.utils import asset_utils, task_registry
import torch

# 初始化Isaac Gym
gym = gymapi.acquire_gym()

# 创建仿真器
sim_params = gymapi.SimParams()
sim_params.dt = 0.01
sim_params.substeps = 1
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

# 设置物理引擎参数
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

# 创建仿真
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("Failed to create sim")
    quit()

# 创建地面
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = 0
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
plane_params.restitution = 0
gym.add_ground(sim, plane_params)

# 创建查看器
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("Failed to create viewer")
    quit()

# 设置相机位置
cam_pos = gymapi.Vec3(2.0, 2.0, 1.5)
cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 创建环境
spacing = 2.0
lower = gymapi.Vec3(-spacing, -spacing, 0.0)
upper = gymapi.Vec3(spacing, spacing, spacing)
num_envs = 1
envs = []

# 加载URDF
# 您需要提供正确的URDF路径，以下是一个示例路径
asset_root = "/home/zhaoqr/devel/unitree_rl_gym_bitbot/resources/robots/bhr8fc2/"  # 修改为你的URDF文件夹路径
urdf_file = "bhr8fc2mc.urdf"  # 修改为你的URDF文件名
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.use_mesh_materials = True
asset_options.flip_visual_attachments = False
asset_options.collapse_fixed_joints = True
asset_options.disable_gravity = False

print(f"Loading URDF: {os.path.join(asset_root, urdf_file)}")
robot_asset = gym.load_asset(sim, asset_root, urdf_file, asset_options)

# 获取关节数量和名称
num_dofs = gym.get_asset_dof_count(robot_asset)
num_bodies = gym.get_asset_rigid_body_count(robot_asset)
print(f"机器人有 {num_dofs} 个自由度和 {num_bodies} 个刚体")

# 打印所有刚体名称
body_names = []
foot_indices = []
# for i in range(num_bodies):
#     name = gym.get_asset_rigid_body_name(robot_asset, i)
#     body_names.append(name)
#     # 自动检测足部链接 (假设足部链接名称包含"foot"或"feet")
#     if "foot" in name.lower() or "feet" in name.lower():
#         foot_indices.append(i)
#     print(f"刚体 {i}: {name}")

# print(f"检测到的足部链接索引: {foot_indices}")
# 创建环境
for i in range(num_envs):
    env = gym.create_env(sim, lower, upper, 1)
    envs.append(env)
    
    # 创建机器人
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.9)  # 初始位置
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # 初始旋转，四元数(w,x,y,z)格式
    
    actor_handle = gym.create_actor(env, robot_asset, pose, "robot", i, 1)
    
    # 设置DOF属性
    props = gym.get_actor_dof_properties(env, actor_handle)
    for j in range(num_dofs):
        props["driveMode"][j] = gymapi.DOF_MODE_POS
        props["stiffness"][j] = 1000.0
        props["damping"][j] = 100.0
        
    gym.set_actor_dof_properties(env, actor_handle, props)

# 设置初始关节位置
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
for i in range(num_envs):
    actor_handle = gym.get_actor_handle(envs[i], 0)
    gym.set_actor_dof_states(envs[i], actor_handle, dof_states, gymapi.STATE_ALL)

# 模拟并获取足部位置
while not gym.query_viewer_has_closed(viewer):
    # 模拟
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    # 更新查看器
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    
    root_state = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_state)
    print("根状态:", root_states[:,2])
    # # 获取足部位置
    # for i in range(num_envs):
    #     actor_handle = gym.get_actor_handle(envs[i], 0)
        
    #     # 输出足部位置
    #     print("足部位置:")
    #     for foot_idx in foot_indices:
    #         rigid_body_state = gym.acquire_rigid_body_state_tensor(sim)
    #         rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
    #         foot_state = rigid_body_states[foot_idx]
    #         foot_name = body_names[foot_idx]
    #         print(f"  {foot_name}: {foot_state}")
    
    # 等待一点时间
    gym.sync_frame_time(sim)

# 清理资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)