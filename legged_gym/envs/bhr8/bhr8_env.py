
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.draw_arrow import WireframeArrowGeometry

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class BHR8Robot(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.debug_viz = False
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg.env.velocity_debug = False
        period = 0.5
        offset = 0.5
        self.phase_offset = torch.rand((self.num_envs,), device=self.device)
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.compute_observations()
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        # period = 0.5
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period + self.phase_offset
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

        self._in_place_flag = torch.norm(self.commands[:, :2], dim=-1) < 0.1
        
        super()._post_physics_step_callback()

        if self.cfg.env.test:
            self.gym.clear_lines(self.viewer)
            self.draw_velocity_actual()
            self.draw_velocity_commanded()
        return 
    
    def compute_ref_state(self):
        phase = self.phase
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = 0.17
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 2] = -sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = sin_pos_l * scale_2
        self.ref_dof_pos[:, 4] = -sin_pos_l * scale_1
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_1
        self.ref_dof_pos[:, 9] = -sin_pos_r * scale_2
        self.ref_dof_pos[:, 10] = sin_pos_r * scale_1
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

        self.ref_action = 2 * self.ref_dof_pos
    
    def step(self, actions):
        # actions += self.ref_action 
        # print(actions)
        return super().step(actions)
    
    def compute_observations(self):
        """ Computes observations
        """
        # self.compute_ref_state()
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        res[self._in_place_flag] = 0
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        res = torch.sum(pos_error, dim=(1))
        res[self._in_place_flag] = 0
        return res
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        res = 1.*single_contact
        res[self._in_place_flag] = 0
        return res

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[0,1,5,6]]), dim=1)

    def _reward_dof_error(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, [5, 11]], dim=1)
    
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target + self.default_dof_pos
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_dist = torch.norm(self.feet_pos[:, 0, :] - self.feet_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)
    
    def _reward_stand_still(self):
        dof_pos_error = torch.norm((self.dof_pos - self.default_dof_pos)[:, :11], dim=1)
        dof_vel_error = torch.norm(self.dof_vel[:, :11], dim=1)
        rew = torch.exp(- 0.1*dof_vel_error) * torch.exp(- dof_pos_error) 
        rew[~self._in_place_flag] = 0
        return rew
############# utils #############

    def draw_velocity_actual(self):
        arrow_head_length = 0.1
        shaft_radius = 0.02
        shaft_segments = 16
        arrow_head_width = shaft_radius * 3
        for i in range(self.num_envs):
            start_point = self.root_states[i, :3].clone()
            # direction = self.base_lin_vel[i, :].clone()
            direction = self.root_states[i, 7:10].clone()
            length = torch.norm(direction)
            arrow = WireframeArrowGeometry(start_point, direction, length, arrow_head_length, arrow_head_width, shaft_radius, shaft_segments, color=(1, 0, 0))
            gymutil.draw_lines(arrow, self.gym, self.viewer, self.envs[i], None)
    
    def draw_velocity_commanded(self):
        arrow_head_length = 0.1
        shaft_radius = 0.02
        shaft_segments = 16
        arrow_head_width = shaft_radius * 3
        direction = self.commands[:, :3].clone()
        direction[:, 2] = 0
        direction = quat_rotate(self.base_quat, direction)
        for i in range(self.num_envs):
            start_point = self.root_states[i, :3].clone()
            direction_i = direction[i, :]
            length = torch.norm(direction_i, dim=-1)
            arrow = WireframeArrowGeometry(start_point, direction_i, length, arrow_head_length, arrow_head_width, shaft_radius, shaft_segments, color=(0, 1, 0))
            gymutil.draw_lines(arrow, self.gym, self.viewer, self.envs[i], None)