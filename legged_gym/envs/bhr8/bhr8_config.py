from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# used for training
class BHR8RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.0] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'lhipYaw' : 0. ,   
           'lhipRoll' : 0.0,               
           'lhipPitch' : -0.3,         
           'lknee' : 0.6,       
           'lankle1' : -0.3,     
           'lankle2' : 0,     
           'rhipYaw' : 0., 
           'rhipRoll' : -0.0, 
           'rhipPitch' : -0.3,                                       
           'rknee' : 0.6,                                             
           'rankle1': -0.3,                              
           'rankle2' : 0,       
        #    'torso_joint' : 0.
        }
    
    class env(LeggedRobotCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-3., 5.]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.8
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hipYaw': 200,
                     'hipRoll': 200,
                     'hipPitch': 300,
                     'knee': 300,
                     'ankle': 60,
                     }  # [N*m/rad]
        damping = {  'hipYaw': 2.5,
                     'hipRoll': 2.5,
                     'hipPitch': 2.5,
                     'knee': 2,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # stiffness = {'hipYaw': 150,
        #              'hipRoll': 150,
        #              'hipPitch': 200,
        #              'knee': 200,
        #              'ankle': 40,
        #              }  # [N*m/rad]
        # damping = {  'hipYaw': 2.5,
        #              'hipRoll': 2.5,
        #              'hipPitch': 2.5,
        #              'knee': 2,
        #              'ankle': 2,
        #              }  # [N*m/rad]  # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/bhr8fc2/bhr8fc2mc.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/bhr8fc2/bhr8fc2mc_add_pos_limit.urdf'
        name = "bhr8"
        foot_name = "foot"
        penalize_contacts_on = ["hip", "calf"]
        terminate_after_contacts_on = ["torso"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.88
        min_dist = 0.2
        max_dist = 0.7
        max_contact_force = 600
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 2.8
            tracking_ang_vel = 1.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 1.0
            collision = 0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.17
            hip_pos = -2.0
            contact_no_vel = -0.8
            feet_swing_height = -20.0
            contact = 0.58
            no_fly = 2
            stand_still_exbody = 1.0
            dof_error = -0.5
            # joint_pos = 3.0
            # feet_distance = 0.0
            # feet_contact_forces = -0.005
            feet_force = -2e-3

# class rewards( LeggedRobotCfg.rewards ):
#     soft_dof_pos_limit = 0.9
#     base_height_target = 0.88
#     min_dist = 0.2
#     max_dist = 0.7
#     max_contact_force = 600
    
    # class scales( LeggedRobotCfg.rewards.scales ):
    #     tracking_lin_vel = 2.8  # 奖励机器人跟踪线速度指令的能力
    #     tracking_ang_vel = 0.8  # 奖励机器人跟踪角速度指令的能力
    #     lin_vel_z = -2.0  # 惩罚机器人在z轴上的线速度偏移
    #     ang_vel_xy = -0.05  # 惩罚机器人在xy平面上的角速度偏移
    #     orientation = -1.0  # 惩罚机器人姿态偏离水平
    #     base_height = -10.0  # 惩罚机器人基座高度偏离目标高度
    #     dof_acc = -2.5e-7  # 惩罚机器人关节加速度过大
    #     dof_vel = -1e-3  # 惩罚机器人关节速度过大
    #     feet_air_time = 1.0  # 奖励机器人足部在空中的时间（长步伐）
    #     collision = 0  # 碰撞惩罚（当前未启用）
    #     action_rate = -0.01  # 惩罚动作变化过快
    #     dof_pos_limits = -5.0  # 惩罚关节位置接近限制
    #     alive = 0.17  # 奖励机器人存活
    #     hip_pos = -2.0  # 惩罚髋关节位置偏移
    #     contact_no_vel = -0.8  # 惩罚落地速度不为零的情况
    #     feet_swing_height = -20.0  # 惩罚足部摆动高度过高
    #     contact = 0.58  # 奖励足部接触地面
    #     no_fly = 2  # 奖励机器人保持足部接触地面（防止飞行）
    #     stand_still_exbody = 1.0  # 奖励机器人保持静止
    #     dof_error = -0.5  # 惩罚特定关节偏离默认位置
    #     feet_force = -1e-3  # 惩罚足部接触力（减小落地冲击）
    #     only_positive_rewards = True

class BHR8RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1.e-3 #5.e-4
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 50000
        run_name = ''
        experiment_name = 'bhr8'

  