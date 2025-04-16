import os
import numpy as np
import xml.etree.ElementTree as ET
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
# from legged_gym.utils import asset_utils, task_registry

def analyze_urdf_mass(urdf_path):
    """
    直接解析URDF文件以获取质量信息
    """
    print(f"直接从URDF文件分析质量信息: {urdf_path}")
    
    try:
        # 解析URDF文件
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        total_mass = 0.0
        links = []
        
        # 遍历所有链接
        for link in root.findall(".//link"):
            link_name = link.get('name')
            inertial = link.find('inertial')
            
            if inertial is not None:
                mass_elem = inertial.find('mass')
                if mass_elem is not None:
                    mass = float(mass_elem.get('value'))
                    total_mass += mass
                    links.append((link_name, mass))
                else:
                    links.append((link_name, 0.0))  # 无质量
            else:
                links.append((link_name, 0.0))  # 无惯性属性
        
        # 打印每个链接的质量
        print("\n--- URDF文件直接分析结果 ---")
        print(f"{'链接名称':<40} {'质量 (kg)':<15}")
        print("-" * 55)
        
        for link_name, mass in links:
            print(f"{link_name:<40} {mass:<15.6f}")
        
        print("-" * 55)
        print(f"{'总质量:':<40} {total_mass:<15.6f} kg")
        
        return links, total_mass
    
    except Exception as e:
        print(f"解析URDF文件时出错: {e}")
        return [], 0.0

def load_urdf_with_isaac_gym(asset_root, urdf_file):
    """
    使用Isaac Gym加载URDF并输出质量信息
    """
    print(f"\n加载URDF文件: {os.path.join(asset_root, urdf_file)}")
    
    # 初始化Isaac Gym
    gym = gymapi.acquire_gym()
    
    # 创建模拟器
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
    
    # 创建模拟
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("Failed to create sim")
        return [], 0.0
    
    # 加载URDF
    asset_options = gymapi.AssetOptions()
    # asset_options.fix_base_link = True
    # asset_options.use_mesh_materials = True
    # asset_options.flip_visual_attachments = False
    # asset_options.collapse_fixed_joints = True
    # asset_options.disable_gravity = True
    
    try:
        robot_asset = gym.load_asset(sim, asset_root, urdf_file, asset_options)
    except Exception as e:
        print(f"加载URDF资产失败: {e}")
        print(f"检查URDF路径: {os.path.join(asset_root, urdf_file)}")
        return [], 0.0
    
    # 获取链接数量和名称
    num_bodies = gym.get_asset_rigid_body_count(robot_asset)
    
    # 创建环境以便获取属性
    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
    
    # 创建机器人actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    
    actor_handle = gym.create_actor(env, robot_asset, pose, "robot", 0, 1)
    
    # 获取链接质量信息
    total_mass = 0.0
    links = []
    
    # 打印每个链接的质量
    print("\n--- Isaac Gym URDF加载结果 ---")
    print(f"{'链接名称':<40} {'质量 (kg)':<15}")
    print("-" * 55)
    
    for i in range(num_bodies):
        body_name = gym.get_asset_rigid_body_name(robot_asset, i)
        props = gym.get_actor_rigid_body_properties(env, actor_handle)
        mass = props[i].mass
        total_mass += mass
        links.append((body_name, mass))
        print(f"{body_name:<40} {mass:<15.6f}")
    
    print("-" * 55)
    print(f"{'总质量:':<40} {total_mass:<15.6f} kg")
    
    # 清理资源
    gym.destroy_sim(sim)
    
    return links, total_mass

def calculate_upper_limb_mass(links):
    """
    计算上肢的总质量
    :param links: 包含链接名称和质量的列表
    :return: 上肢总质量
    """
    upper_limb_mass = 0.0
    print("\n--- 上肢质量计算 ---")
    print(f"{'链接名称':<40} {'质量 (kg)':<15}")
    print("-" * 55)
    
    for link_name, mass in links:
        # 假设上肢的链接名称包含 "arm" 或 "upper"
        if "arm" in link_name.lower() or "shoulder" in link_name.lower() or "torso" in link_name.lower():
            upper_limb_mass += mass
            print(f"{link_name:<40} {mass:<15.6f}")
    
    print("-" * 55)
    print(f"{'上肢总质量:':<40} {upper_limb_mass:<15.6f} kg")
    return upper_limb_mass

def main():
    """
    主函数，用于处理命令行参数并运行分析
    """
    # 获取命令行参数
    # 您需要提供正确的URDF路径
    # args = gymutil.parse_arguments(
    #     description="URDF质量分析工具",
    #     custom_parameters=[
    #         {"name": "--urdf_file", "type": str, "default": "robot.urdf", "help": "URDF文件名"},
    #         {"name": "--asset_root", "type": str, "default": os.getcwd(), "help": "URDF文件所在目录路径"}
    #     ]
    # )
    asset_root = "/home/zhaoqr/devel/unitree_rl_gym_bitbot/resources/robots/bhr8fc2/"  # 修改为你的URDF文件夹路径
    urdf_file = "bhr8fc2mc.urdf"  # 修改为你的URDF文件名
    
    # 打印参数信息
    print(f"\n--- URDF质量分析 ---")
    print(f"URDF文件路径: {os.path.join(asset_root, urdf_file)}")
    
    # 直接解析URDF文件获取质量信息
    urdf_links, urdf_total_mass = analyze_urdf_mass(os.path.join(asset_root, urdf_file))
    
    # 使用Isaac Gym加载URDF并获取质量信息
    isaac_links, isaac_total_mass = load_urdf_with_isaac_gym(asset_root, urdf_file)
    
    # 比较两种方法的结果
    if urdf_total_mass > 0 and isaac_total_mass > 0:
        diff = abs(urdf_total_mass - isaac_total_mass)
        print(f"\n两种方法的总质量差异: {diff:.6f} kg ({(diff/urdf_total_mass)*100:.2f}%)")
    
    calculate_upper_limb_mass(isaac_links)
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()