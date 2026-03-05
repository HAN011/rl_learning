from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz, torch_rand_float

import torch
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg
from humanoid.envs.custom.humanoid_env import XBotLFreeEnv


class TwoWheelBalancerEnv(XBotLFreeEnv):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        tilt_rad = self.cfg.env.max_tilt_deg * torch.pi / 180.0
        self.max_tilt_cos = torch.cos(torch.tensor(tilt_rad, device=self.device))

    def _compute_torques(self, actions):
        torques = actions * self.cfg.control.action_scale
        torques -= self.cfg.control.wheel_damping * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def check_termination(self):
        base_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if len(self.termination_contact_indices) > 0:
            base_contact = torch.any(
                torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
                dim=1,
            )

        upright_cos = -self.projected_gravity[:, 2]
        over_tilt = upright_cos < self.max_tilt_cos
        low_base = self.root_states[:, 2] < self.cfg.env.min_base_height

        self.reset_buf = base_contact | over_tilt | low_base
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def _reset_dofs(self, env_ids):
        if len(env_ids) == 0:
            return
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(
            -0.05, 0.05, (len(env_ids), self.num_dof), device=self.device
        )
        max_wheel_vel = self.cfg.env.reset_wheel_vel
        self.dof_vel[env_ids] = torch_rand_float(
            -max_wheel_vel, max_wheel_vel, (len(env_ids), self.num_dof), device=self.device
        )

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        if len(env_ids) == 0:
            return
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]

        max_angle = self.cfg.env.reset_angle_range
        roll = torch_rand_float(-max_angle, max_angle, (len(env_ids), 1), device=self.device).squeeze(1)
        pitch = torch_rand_float(-max_angle, max_angle, (len(env_ids), 1), device=self.device).squeeze(1)
        yaw = torch_rand_float(-0.05, 0.05, (len(env_ids), 1), device=self.device).squeeze(1)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

        self.root_states[env_ids, 7:10] = torch_rand_float(-0.2, 0.2, (len(env_ids), 3), device=self.device)
        self.root_states[env_ids, 10:13] = torch_rand_float(-0.4, 0.4, (len(env_ids), 3), device=self.device)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reward_upright(self):
        tilt_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return torch.exp(-self.cfg.rewards.upright_sigma * tilt_error)

    def _reward_stability(self):
        pitch_rate = self.base_ang_vel[:, 1]
        return torch.exp(-self.cfg.rewards.stability_sigma * torch.square(pitch_rate))

    def _reward_energy(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_position(self):
        drift_xy = self.root_states[:, :2] - self.env_origins[:, :2]
        return torch.sum(torch.square(drift_xy), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_termination(self):
        return (self.reset_buf.bool() & (~self.time_out_buf)).float()
