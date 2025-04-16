import os
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.terrain_utils import *
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

# 创建粗糙地面
def create_rough_terrain(sim):
    # 使用您提供的参数
    mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
    horizontal_scale = 0.1  # [m]
    vertical_scale = 0.005  # [m]
    border_size = 25  # [m]
    static_friction = 1.0
    dynamic_friction = 1.0
    restitution = 0.0
    
    # 粗糙地形的测量点
    measure_heights = {'type': 'random_uniform_terrain', 'min_height': -0.15, 'max_height': 0.15, 'step': 0.01, 'downsampled_scale': 0.2}
    measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # 粗糙地形参数
    terrain_width = 50  # 地形宽度
    terrain_length = 50  # 地形长度
    x_size = int(terrain_width / horizontal_scale)
    y_size = int(terrain_length / horizontal_scale)
    
    # 从measure_heights参数中获取值
    min_height = measure_heights['min_height']
    max_height = measure_heights['max_height']
    step = measure_heights['step']
    downsampled_scale = measure_heights['downsampled_scale']
    
    print(f"生成粗糙地形: 类型={mesh_type}, 高度范围=[{min_height}, {max_height}]m")
    
    # 生成随机均匀地形高度场
    heightfield = np.zeros((x_size, y_size), dtype=np.float32)
    
    # 使用随机均匀地形生成方法
    # 首先生成一个低分辨率的随机地形
    ds_scale = downsampled_scale / horizontal_scale
    ds_x_size = int(x_size * ds_scale)
    ds_y_size = int(y_size * ds_scale)
    
    # 生成低分辨率随机高度场
    ds_heightfield = np.random.uniform(
        low=min_height / vertical_scale,
        high=max_height / vertical_scale,
        size=(ds_x_size, ds_y_size)
    ).astype(np.float32)
    
    # 上采样到目标分辨率
    from scipy.ndimage import zoom
    zoom_factor = 1.0 / ds_scale
    heightfield = zoom(ds_heightfield, zoom_factor, order=1)
    
    # 裁剪到正确的尺寸
    heightfield = heightfield[:x_size, :y_size]
    
    # 从高度值转换为实际高度（以米为单位）
    heightfield = heightfield * vertical_scale
    
    # 创建地形
    terrain_params = gymapi.TriangleMeshParams()
    terrain_params.nb_vertices = x_size * y_size
    terrain_params.nb_triangles = 2 * (x_size - 1) * (y_size - 1)
    terrain_params.transform.p.x = -terrain_width / 2.0
    terrain_params.transform.p.y = -terrain_length / 2.0
    terrain_params.transform.p.z = 0.0
    terrain_params.static_friction = static_friction
    terrain_params.dynamic_friction = dynamic_friction
    terrain_params.restitution = restitution
    
    # 创建顶点和三角形数组
    vertices = np.zeros((x_size * y_size, 3), dtype=np.float32)
    triangles = np.zeros((2 * (x_size - 1) * (y_size - 1), 3), dtype=np.int32)
    
    # 填充顶点数据
    for i in range(x_size):
        for j in range(y_size):
            vertices[i * y_size + j, 0] = i * horizontal_scale
            vertices[i * y_size + j, 1] = j * horizontal_scale
            vertices[i * y_size + j, 2] = heightfield[i, j]
    
    # 填充三角形索引
    triangle_idx = 0
    for i in range(x_size - 1):
        for j in range(y_size - 1):
            triangles[triangle_idx, 0] = i * y_size + j
            triangles[triangle_idx, 1] = i * y_size + j + 1
            triangles[triangle_idx, 2] = (i + 1) * y_size + j
            triangle_idx += 1
            
            triangles[triangle_idx, 0] = (i + 1) * y_size + j
            triangles[triangle_idx, 1] = i * y_size + j + 1
            triangles[triangle_idx, 2] = (i + 1) * y_size + j + 1
            triangle_idx += 1
    
    # 创建网格资产
    terrain_asset = gym.create_triangle_mesh(
        sim,
        vertices.flatten(),
        triangles.flatten(),
        terrain_params
    )
    
    print(f"创建了粗糙地形，顶点数: {x_size * y_size}, 三角形数: {2 * (x_size - 1) * (y_size - 1)}")
    
    return terrain_asset

# 生成并添加粗糙地形
terrain_asset = create_rough_terrain(sim)

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
for i in range(num_bodies):
    name = gym.get_asset_rigid_body_name(robot_asset, i)
    body_names.append(name)
    # 自动检测足部链接 (假设足部链接名称包含"foot"或"feet")
    if "foot" in name.lower() or "feet" in name.lower():
        foot_indices.append(i)
    print(f"刚体 {i}: {name}")

print(f"检测到的足部链接索引: {foot_indices}")

# 创建环境
for i in range(num_envs):
    env = gym.create_env(sim, lower, upper, 1)
    envs.append(env)
    
    # 添加地形到环境
    terrain_pose = gymapi.Transform()
    terrain_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    terrain_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    terrain_handle = gym.create_actor(env, terrain_asset, terrain_pose, "terrain", i, 0)
    
    # 创建机器人
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 1)  # 初始位置
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
    
    # 获取足部位置
    for i in range(num_envs):
        actor_handle = gym.get_actor_handle(envs[i], 0)
        
        # 输出足部位置
        print("足部位置:")
        for foot_idx in foot_indices:
            foot_transform = gym.get_rigid_transform(envs[i], gym.get_actor_rigid_body_handle(envs[i], actor_handle, foot_idx))
            foot_pos = [foot_transform.p.x, foot_transform.p.y, foot_transform.p.z]
            foot_name = body_names[foot_idx]
            print(f"  {foot_name}: {foot_pos}")
            
        # 测量足部下方的地形高度
        if len(foot_indices) > 0:
            print("\n测量地形高度点:")
            measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
            
            robot_transform = gym.get_actor_rigid_body_transform(envs[i], actor_handle, 0)  # 获取机器人根节点的变换
            
            # 示例：只打印部分点的高度，避免输出过多数据
            for x in measured_points_x[::4]:  # 每隔4个点取一个
                for y in measured_points_y[::4]:  # 每隔4个点取一个
                    # 将点从机器人局部坐标转换到世界坐标
                    world_x = robot_transform.p.x + x
                    world_y = robot_transform.p.y + y
                    
                    # 使用光线投射来获取地形高度
                    raycast_data = gymapi.RaycastArgs()
                    raycast_data.start.x = world_x
                    raycast_data.start.y = world_y
                    raycast_data.start.z = 2.0  # 从足够高的位置开始投射
                    raycast_data.end.x = world_x
                    raycast_data.end.y = world_y
                    raycast_data.end.z = -2.0  # 到足够低的位置结束
                    
                    hit = gym.raycast(sim, envs[i], raycast_data)
                    if hit:
                        print(f"  点 ({x:.2f}, {y:.2f}) 处的地形高度: {raycast_data.hit.z:.3f}")
    
    # 等待一点时间
    gym.sync_frame_time(sim)

# 清理资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)