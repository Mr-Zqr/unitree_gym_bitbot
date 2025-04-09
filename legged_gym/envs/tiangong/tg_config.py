from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TGConfig( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.85] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'ankle_pitch_l_joint' : -0.2,
            'ankle_pitch_r_joint' : -0.2,
            'ankle_roll_l_joint' : 0.0,
            'ankle_roll_r_joint' : 0.0,
            'elbow_l_joint' : 0.0,
            'elbow_r_joint' : 0.0,
            'hip_pitch_l_joint' : -0.3,
            'hip_pitch_r_joint' : -0.3,
            'hip_roll_l_joint' : 0.0,
            'hip_roll_r_joint' : 0.0,
            'hip_yaw_l_joint' : 0.0,
            'hip_yaw_r_joint' : 0.0,
            'knee_pitch_l_joint' : 0.5,
            'knee_pitch_r_joint' : 0.5,
            'shoulder_pitch_l_joint' : 0.0,
            'shoulder_pitch_r_joint' : 0.0,
            'shoulder_roll_l_joint' : 0.0,
            'shoulder_roll_r_joint' : 0.0,
            'shoulder_yaw_l_joint' : 0.0,
            'shoulder_yaw_r_joint' : 0.0,
        }
    
    class env(LeggedRobotCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 150,
                     'hip_roll': 150,
                     'hip_pitch': 150,
                     'knee': 200,
                     'ankle': 40,
                     'shoulder': 150,
                     'elbow': 100,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'shoulder': 2,
                     'elbow': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/tiangong/tiangong_modified_collision.urdf'
        name = "tiangong"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.80
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 1.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.85
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.48
            dof_error = -0.3

class TGConfigPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [ 64, 32]
        critic_hidden_dims = [ 64, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 64
        # rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.008
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCritic"
        max_iterations = 70000
        run_name = ''
        experiment_name = 'tiangong'

  
