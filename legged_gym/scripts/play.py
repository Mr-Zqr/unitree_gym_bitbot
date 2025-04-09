import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.vel_debug:
        env_cfg.env.num_envs = 16
        env_cfg.env.velocity_debug = True
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 32)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    print("****************")
    print(env.dof_names)
    # init logger
    if args.show_log:
        dof_names = env.dof_names
        dof_pos_limits = env.dof_pos_limits
        logger = Logger(env.dt, dof_names)
        robot_idx = 0
        stop_record_log = 150
        action_scale = env_cfg.control.action_scale

        # get the dir to loaded run and model: 
        log_root_plt = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
        runs = os.listdir(log_root_plt)
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        loaded_run = -1
        model_plt = -1
        if train_cfg.runner.load_run == -1:
            loaded_run = runs[-1]
        else:
            loaded_run = train_cfg.runner.load_run
        # get the last checkpoint in dir last_run
        if train_cfg.runner.checkpoint == -1:
            models_plt = [file for file in os.listdir(os.path.join(log_root_plt, loaded_run)) if 'model' in file]
            models_plt.sort(key=lambda m: '{0:0>15}'.format(m))
            model_plt = models_plt[-1].split('.')[0]
        else:
            model_plt  = f'model_{train_cfg.runner.checkpoint}'
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        if args.show_log:
            if i < stop_record_log:
                logger.log_joint_state(robot_index=robot_idx, dof_pos=env.dof_pos, action=actions, ref_pos=None)
            elif i == stop_record_log:
                logger.plot_states(train_cfg.runner.experiment_name, loaded_run, model_plt , dof_pos_limits, log_root_plt, action_scale)    

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
