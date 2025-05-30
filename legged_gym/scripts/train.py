import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils import ZzsExperimentLogger
from legged_gym.utils.helpers import class_to_dict
import torch

def train(args):
    print("Training")
    logdir = ZzsExperimentLogger.generate_logdir(args.task, datetime.now().strftime('%b%d_%H-%M'))
    exp_msg = {}
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    exp_msg["env_cfg"] = class_to_dict(env_cfg)
    exp_msg["train_cfg"] = class_to_dict(train_cfg)
    ZzsExperimentLogger.save_hyper_params(logdir, env_cfg, train_cfg)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
