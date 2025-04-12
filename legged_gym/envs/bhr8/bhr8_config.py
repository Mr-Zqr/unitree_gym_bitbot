from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

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
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.0
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        # stiffness = {'hipYaw': 200,
        #              'hipRoll': 200,
        #              'hipPitch': 200,
        #              'knee': 300,
        #              'ankle': 70,
        #              }  # [N*m/rad]
        # damping = {  'hipYaw': 3,
        #              'hipRoll': 3,
        #              'hipPitch': 3,
        #              'knee': 4,
        #              'ankle': 3,
        #              }  # [N*m/rad]  # [N*m*s/rad]
        stiffness = {'hipYaw': 150,
                     'hipRoll': 150,
                     'hipPitch': 200,
                     'knee': 200,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hipYaw': 2.5,
                     'hipRoll': 2.5,
                     'hipPitch': 2.5,
                     'knee': 2,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/bhr8fc2/bhr8fc2mc.urdf'
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
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -4.0
            dof_acc = -2.5e-7
            dof_vel = -0.0001
            feet_air_time = 1.0
            collision = -1.0
            action_rate = -0.05
            dof_pos_limits = -5.0
            alive = 0.56
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.38
            no_fly = 2
            stand_still_exbody = 5
            dof_error = 0.0
            # joint_pos = 3.0
            feet_distance = 0.0
            # feet_contact_forces = -0.01
            feet_force = -3e-3

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

  
