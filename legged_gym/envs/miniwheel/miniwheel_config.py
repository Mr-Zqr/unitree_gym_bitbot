from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class MiniWheelRoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.82] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "lhiproll": 0.0,
            "lfempitch": 0.3,
            "ltibpitch": -0.6,
            "lfootrot": -0.3,
            "rhiproll": 0.0,
            "rfempitch": 0.3,
            "rtibpitch": -0.6,
            "rfootrot": -0.3,
        }
    
    class env(LeggedRobotCfg.env):
        num_observations = 35
        num_privileged_obs = 38
        num_actions = 8


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {"hiproll": 60, "fempitch": 60, "tibpitch": 100, "footrot": 20}  # P
        damping = {
            "hiproll": 1.5,
            "fempitch": 1.5,
            "tibpitch": 2.5,
            "footrot": 0.5,
        }  # [N*m/rad] D
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/BIT_miniwheel/urdf/BITwheel_foot.urdf'
        name = "miniwheel"
        foot_name = "foot"
        penalize_contacts_on = ["fem", "tib", "foot"]
        terminate_after_contacts_on = ["torso"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.82
        min_dist = 0.2
        max_dist = 0.7
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -4.0
            dof_acc = -1e-6
            dof_vel = -5e-3
            feet_air_time = 1.0
            collision = -1.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.16
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.38
            no_fly = 2
            # dof_error = -0.5
            # joint_pos = 3.0
            feet_distance = 0.0

class MiniWheelRoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [16]
        critic_hidden_dims = [16]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.001
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'miniwheel'

  
