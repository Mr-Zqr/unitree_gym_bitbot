import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

class Logger:
    _joint_states = {}
    rad2deg = 180 / np.pi
    def __init__(self, dt, dof_names):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None
        self.dof_names = dof_names

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def _log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)
    
    def log_joint_state(self, robot_index, dof_pos, action, ref_pos=None):
        dof_pos = dof_pos.cpu().detach().numpy()
        action = action.cpu().detach().numpy()
        for i in range(dof_pos.shape[1]):
            self._joint_states[f'measured_{i}'] = dof_pos[robot_index, i]
            self._joint_states[f'policy_action_{i}'] = action[robot_index, i]
        self._log_states(self._joint_states)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def plot_states(self, experiment_name, loaded_run, checkpoint, dof_pos_limits, log_root, action_scale):
        if self.plot_process is None:
            dof_pos_limits = dof_pos_limits.cpu().detach().numpy()
            self.plot_process = Process(target=self._plot, args=(experiment_name, loaded_run, checkpoint, dof_pos_limits, log_root, action_scale))
            self.plot_process.start()
    
    def _plot(self, experiment_name, loaded_run, checkpoint, dof_pos_limits, log_root, action_scale):
            import matplotlib.pyplot as plt
            import os

            # Determine the number of subplots
            num_pairs = len([key for key in self.state_log.keys() if key.startswith('measured_')])
            num_cols = int(np.ceil(np.sqrt(num_pairs)))
            num_rows = int(np.ceil(num_pairs / num_cols))

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
            axes = axes.flatten()

            for i in range(num_pairs):
                measured_key = f'measured_{i}'
                action_key = f'policy_action_{i}'
                if measured_key in self.state_log and action_key in self.state_log:
                    time = np.linspace(0, self.dt * len(self.state_log[measured_key]), len(self.state_log[measured_key]))
                    measured_values = np.array(self.state_log[measured_key])*self.rad2deg
                    action_values = np.array(self.state_log[action_key])*self.rad2deg*action_scale
                    axes[i].plot(time, measured_values, label='Measured')
                    axes[i].plot(time, action_values, label='Policy Action')

                    lower_limit, upper_limit = dof_pos_limits[i] * self.rad2deg
                    axes[i].axhline(lower_limit, color='r', linestyle='--', label='Lower Limit')
                    axes[i].axhline(upper_limit, color='r', linestyle='--', label='Upper Limit')
                    # axes[i].set_ylim([lower_limit*1.20, upper_limit*1.20])

                    joint_name = self.dof_names[i]
                    axes[i].set_title(joint_name)

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower right')
            title = f'{experiment_name}-{loaded_run}-{checkpoint}'
            plt.suptitle(title)
            plt.tight_layout()

            # save fig
            save_dir = os.path.join(log_root, loaded_run, 'log_plots')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'joint_states_{title}.png')
            plt.savefig(save_path)
            plt.show()
    
    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()   

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
