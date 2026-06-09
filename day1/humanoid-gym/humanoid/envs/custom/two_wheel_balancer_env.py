from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz, torch_rand_float

import torch
from humanoid.envs.base.legged_robot import LeggedRobot
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg
from humanoid.utils.terrain import HumanoidTerrain


class TwoWheelBalancerEnv(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        tilt_rad = self.cfg.env.max_tilt_deg * torch.pi / 180.0
        self.max_tilt_cos = torch.cos(torch.tensor(tilt_rad, device=self.device))
        self.ref_action = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)
        self.pitch_int = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.over_tilt_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params
        )
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _get_phase(self):
        cycle_time = getattr(self.cfg.rewards, "cycle_time", 1.0)
        phase = self.phase_length_buf * self.dt / cycle_time
        return phase

    def step(self, actions):
        if self.cfg.env.use_ref_actions:
            residual_scale = float(getattr(self.cfg.env, "residual_action_scale", 1.0))
            self._compute_ref_actions()
            actions = residual_scale * actions + self.ref_action

        actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions
        return super().step(actions)

    def _compute_ref_actions(self):
        gains = self.cfg.env
        # projected_gravity in body frame: upright is [0, 0, -1]
        pitch = torch.atan2(self.projected_gravity[:, 0], -self.projected_gravity[:, 2])
        pitch_rate = self.base_ang_vel[:, 1]
        x_err = self.root_states[:, 0] - self.env_origins[:, 0]
        x_vel = self.root_states[:, 7]

        self.pitch_int = torch.clip(
            self.pitch_int + pitch * self.dt,
            -float(gains.ref_pitch_i_clip),
            float(gains.ref_pitch_i_clip),
        )

        u = (
            float(gains.ref_kp_pitch) * pitch
            - float(gains.ref_kd_pitch) * pitch_rate
            + float(gains.ref_ki_pitch) * self.pitch_int
            - float(gains.ref_kp_pos) * x_err
            - float(gains.ref_kd_pos) * x_vel
        )
        u = torch.clip(u, -float(gains.ref_action_clip), float(gains.ref_action_clip))
        u = -u
        self.ref_action[:, 0] = u
        self.ref_action[:, 1] = u

    def _compute_torques(self, actions):
        torques = actions * self.cfg.control.action_scale
        torques -= self.cfg.control.wheel_damping * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def check_termination(self):
        base_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if len(self.termination_contact_indices) > 0:
            contact_thresh = float(self.cfg.env.base_contact_force_threshold)
            base_contact = torch.any(
                torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > contact_thresh,
                dim=1,
            )

        upright_cos = -self.projected_gravity[:, 2]
        over_tilt_raw = upright_cos < self.max_tilt_cos
        grace_steps = int(getattr(self.cfg.env, "over_tilt_grace_steps", 1))
        self.over_tilt_count = torch.where(
            over_tilt_raw,
            self.over_tilt_count + 1,
            torch.zeros_like(self.over_tilt_count),
        )
        over_tilt = self.over_tilt_count >= grace_steps
        low_base = self.root_states[:, 2] < self.cfg.env.min_base_height

        self.reset_buf = base_contact | over_tilt | low_base
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf
        self._base_height_at_check = self.root_states[:, 2].clone()
        self._reset_reason_masks = {
            "base_contact": base_contact,
            "over_tilt": over_tilt,
            "over_tilt_raw": over_tilt_raw,
            "low_base": low_base,
            "time_out": self.time_out_buf.clone(),
        }

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

        max_lin_vel = float(self.cfg.env.reset_base_lin_vel)
        max_ang_vel = float(self.cfg.env.reset_base_ang_vel)
        self.root_states[env_ids, 7:10] = torch_rand_float(
            -max_lin_vel, max_lin_vel, (len(env_ids), 3), device=self.device
        )
        self.root_states[env_ids, 10:13] = torch_rand_float(
            -max_ang_vel, max_ang_vel, (len(env_ids), 3), device=self.device
        )

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def get_obs_noise(self, name):
        if name in ["command_input", "actions"]:
            return torch.zeros((self.num_envs, self.cfg.env.get_obs_dim(name)), device=self.device)
        if name in ["dof_pos", "dof_vel", "actions", "base_lin_vel", "base_ang_vel"]:
            scale = getattr(self.cfg.noise.noise_scales, name)
            return torch.randn((self.num_envs, self.cfg.env.get_obs_dim(name)), device=self.device) * scale
        if name.startswith("base_euler"):
            scale = getattr(self.cfg.noise.noise_scales, "base_euler")
            return torch.randn((self.num_envs, self.cfg.env.get_obs_dim(name)), device=self.device) * scale
        raise NotImplemented

    def compute_observations(self):
        privileged_obs_now = torch.cat([self.get_obs(name) for name in self.cfg.env.privileged_obs_names], dim=-1)
        obs_now = torch.cat([self.get_obs(name) for name in self.cfg.env.obs_names], dim=-1)
        if self.cfg.noise.add_noise:
            noise = torch.cat([self.get_obs_noise(name) for name in self.cfg.env.obs_names], dim=-1)
            obs_now += noise * self.cfg.noise.noise_level

        self.obs_history.append(obs_now)
        self.critic_history.append(privileged_obs_now)
        self.obs_buf = torch.cat([self.obs_history[i] for i in range(self.cfg.env.frame_stack)], dim=1)
        self.privileged_obs_buf = torch.cat(
            [self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) > 0 and hasattr(self, "_reset_reason_masks"):
            reason_masks = self._reset_reason_masks
            self.extras["episode"]["reset_base_contact_ratio"] = reason_masks["base_contact"][env_ids].float().mean()
            self.extras["episode"]["reset_over_tilt_ratio"] = reason_masks["over_tilt"][env_ids].float().mean()
            self.extras["episode"]["reset_over_tilt_raw_ratio"] = reason_masks["over_tilt_raw"][env_ids].float().mean()
            self.extras["episode"]["reset_low_base_ratio"] = reason_masks["low_base"][env_ids].float().mean()
            self.extras["episode"]["reset_timeout_ratio"] = reason_masks["time_out"][env_ids].float().mean()
            if hasattr(self, "_base_height_at_check"):
                self.extras["episode"]["reset_base_height_mean"] = self._base_height_at_check[env_ids].mean()
        if len(env_ids) > 0:
            self.pitch_int[env_ids] = 0.0
            self.over_tilt_count[env_ids] = 0
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

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

    def _reward_alive(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_termination(self):
        return (self.reset_buf.bool() & (~self.time_out_buf)).float()
