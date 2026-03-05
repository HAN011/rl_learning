# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import os
import numpy as np
import xml.etree.ElementTree as ET

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from collections import deque

import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs.base.base_task import BaseTask
# from humanoid.utils.terrain import Terrain
from humanoid.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from humanoid.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
import time

# def get_euler_xyz_tensor(quat):
#     r, p, w = get_euler_xyz(quat)
#     # stack r, p, w in dim1
#     euler_xyz = torch.stack((r, p, w), dim=1)
#     euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
#     return euler_xyz
def copysign_new(a, b):

    a = torch.tensor(a, device=b.device, dtype=torch.float)
    a = a.expand_as(b)
    return torch.abs(a) * torch.sign(b)

def get_euler_rpy(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = q[..., qw] * q[..., qw] - q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] + q[..., qz] * q[..., qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign_new(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = q[..., qw] * q[..., qw] + q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] - q[..., qz] * q[..., qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_rpy(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=-1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.asset_type = None
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()#self.create_sim()在上面的 super().__init__中调用
        self._prepare_reward_function()
        self.init_done = True
        #self.start_time = time.time()

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        #self.end_time = time.time()
        #print(self.end_time - self.start_time)
        #self.start_time=self.end_time

        clip_actions = self.cfg.normalization.clip_actions # 100. 
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)# 限制动作范围
        # step physics and render each frame
        self.render()#. 渲染当前画面（如果开启 viewer）
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            #将力矩张量绑定到仿真器的关节控制输入上
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            # 3.3 执行物理模拟一步
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
             # 3.5 刷新自由度的状态张量（位置、速度）
            self.gym.refresh_dof_state_tensor(self.sim)
        #每个self.dt=0.001*10调用下面这个函数
        
        self.post_physics_step()#
        
        #print(self.root_states[:, :])这个和下面这个相等
        #print(self.rigid_state[:, 0, :])
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)#self.obs_buf这个在上面的post_physics_step函数里更新
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        #self.rew_buf这个在上面的post_physics_step函数里的compute_reward函数更新
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras


    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        ## 刷新 root_states（位姿+速度）
        self.gym.refresh_actor_root_state_tensor(self.sim)
        ## 刷新接触力张量（用于判断触地/接触）
        self.gym.refresh_net_contact_force_tensor(self.sim)
        ## 刷新每个刚体的状态（位置、姿态、速度等）
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        #     ===== 2. 累加步长 =====
        
        self.episode_length_buf += 1    # 每个环境的 episode 步数 +1   记录每个环境当前已经运行了多少个时间步。
        self.common_step_counter += 1   # 所有环境共享的总步数 +1
        
        #print(self.episode_length_buf)
        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        # 以下通过逆旋转，将速度从世界系变换到机体系
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # 重力向量投影到机体坐标系（通常是 [0, 0, -9.81] 旋转到机体系）
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        #self.feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        #self.feet_euler_xyz = get_euler_xyz_tensor(self.feet_quat)
        #print(self.feet_euler_xyz)
        #下面这函数核心调用核心调用self._resample_commands()，更新self.commands[:, ]
        self._post_physics_step_callback()#里面会执行 self._resample_commands(env_ids)

        # compute observations, rewards, resets, ...
        self.check_termination() # 判断是否满足 episode 终止条件（如摔倒）
        self.compute_reward() # 奖励函数计算
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        #print(self.reset_buf)
        #print(env_ids)
        #假如5个环境，输出
        #tensor([False, False,  True, False, False], device='cuda:0')
        #tensor([2], device='cuda:0')  self.reset_buf.nonzero(as_tuple=False)是tensor([[1]])，indices.flatten()是tensor([1])
        self.reset_idx(env_ids)# 重置选中的环境 )#里面会执行 self._resample_commands(env_ids)
        # 更新观测张量（例如位置、速度、命令等）
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]
        # ===== 7. 可视化调试（例如绘制力、接触点、坐标系等）=====
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        #torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1)是求每个link的xyz平均力
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        #print(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1))#假设有两个环境体输出tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0')
        #print(self.termination_contact_indices)#输出tensor([ 0, 12, 13, 14, 15, 16, 17, 18, 19], device='cuda:0')
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
       
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
       
        # reset robot states
        self._reset_dofs(env_ids)

        self._reset_root_states(env_ids)

        #self._resample_commands(env_ids)

        # reset buffers
        self.last_last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_rigid_state[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.phase_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        self.gait_start[env_ids] = torch.randint(0, 2, (len(env_ids),)).to(self.device)*0.5  
        #resample command
        self.generate_gait_time(env_ids)
        self._resample_commands()

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
            
        # fix reset gravity bug
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        #self.feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        #self.feet_euler_xyz = get_euler_xyz_tensor(self.feet_quat)

    def generate_gait_time(self,envs):
        if len(envs) == 0:
            return

        # rand sample 
        random_tensor_list = []
        for i in range(len(self.cfg.commands.gait)):
            name = self.cfg.commands.gait[i]
            gait_time_range = self.cfg.commands.gait_time_range[name]
            random_tensor_single = torch_rand_float(gait_time_range[0],
                                            gait_time_range[1],
                                            (len(envs), 1),device=self.device)
            random_tensor_list.append(random_tensor_single)

        random_tensor = torch.cat([random_tensor_list[i] for i in range(len(self.cfg.commands.gait))], dim=1)
        current_sum = torch.sum(random_tensor,dim=1,keepdim=True)
        # scaled_tensor store proportion for each gait type
        scaled_tensor = random_tensor * (self.max_episode_length / current_sum)
        scaled_tensor[:,1:] = scaled_tensor[:,:-1].clone()
        scaled_tensor[:,0] *= 0.0
        #print(self.max_episode_length)
        #print(random_tensor_list)
        #print(random_tensor)
        #print(current_sum)
        # self.gait_time accumulate gait_duration_tick
        # self.gait_time = |__gait1__|__gait2__|__gait3__|
        # self.gait_time triger resample gait command
        self.gait_time[envs] = torch.cumsum(scaled_tensor,dim=1).int()

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        
    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        #print(len(props))输出20
        for s in range(len(props)):
            if self.cfg.domain_rand.randomize_friction:
                props[s].friction = self.friction_coeffs[env_id]
            if self.cfg.domain_rand.randomize_restitution:
                props[s].restitution = self.restitution_coeffs[env_id]
        return props
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.original_friction = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.original_armature = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item() * self.cfg.safety.pos_limit
                self.dof_pos_limits[i, 1] = props["upper"][i].item() * self.cfg.safety.pos_limit
                self.dof_vel_limits[i] = props["velocity"][i].item() * self.cfg.safety.vel_limit
                self.torque_limits[i] = props["effort"][i].item() * self.cfg.safety.torque_limit
                self.original_friction[i] = props["friction"][i].item()
                self.original_armature[i] = props["armature"][i].item()
        
        for i in range(len(props)):
            if self.cfg.domain_rand.randomize_joint_friction:
                props["friction"][i] = (self.original_friction[i] * self.joint_friction_coeffs[env_id, i]).item()
            if self.cfg.domain_rand.randomize_joint_armature:
                props["armature"][i] = (self.original_armature[i] * self.joint_armature_coeffs[env_id, i]).item()

        return props
    
    def _process_rigid_body_props(self, props, env_id):
        if env_id==0:
            self.original_base_mass = props[0].mass
            self.original_base_com = [props[0].com.x, props[0].com.y, props[0].com.z]
            # self.original_base_mass = props[11].mass
            # self.original_base_com = [props[11].com.x, props[11].com.y, props[11].com.z]
        # for i, prop in enumerate(props):
        #     print(f"Body {i}: com.x = {prop.com.x:.6f}, com.y = {prop.com.y:.6f}, com.z = {prop.com.z:.6f}")

        #base
        if self.cfg.domain_rand.randomize_base_mass:
            props[0].mass = self.original_base_mass + self.base_mass_coeffs[env_id].item()
            #props[11].mass = self.original_base_mass + self.base_mass_coeffs[env_id].item()
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = gymapi.Vec3(
                self.original_base_com[0] + self.base_com_coeffs[env_id, 0].item(),
                self.original_base_com[1] + self.base_com_coeffs[env_id, 1].item(),
                self.original_base_com[2] + self.base_com_coeffs[env_id, 2].item())
            
        # if self.cfg.domain_rand.randomize_base_inertia:
        #     props[0].inertia.x.x *= self.base_inertia_x[env_id]
        #     props[0].inertia.y.y *= self.base_inertia_y[env_id]
        #     props[0].inertia.z.z *= self.base_inertia_z[env_id]

        #link
        # if self.cfg.domain_rand.randomize_link_mass:
        #     for i in range(1, len(props)):
        #         props[i].mass *= self.link_masses[env_id, i-1]

        # if self.cfg.domain_rand.randomize_link_com:
        #     for i in range(1, len(props)):
        #         props[i].com = gymapi.Vec3(
        #             props[i].com.x+=self.link_com_coeffs[env_id, 0].item(), 
        #             props[i].com.y+=self.link_com_coeffs[env_id, 1].item(),
        #             props[i].com.z+=self.link_com_coeffs[env_id, 2].item()) 
                
        # if self.cfg.domain_rand.randomize_link_inertia:
        #     for i in range(1, len(props)):
        #         props[i].inertia.x.x *= self.link_inertia_x[env_id]
        #         props[i].inertia.y.y *= self.link_inertia_y[env_id]
        #         props[i].inertia.z.z *= self.link_inertia_z[env_id]

        return props
    
    # def _post_physics_step_callback(self):
    #     """ Callback called before computing terminations, rewards, and observations
    #         Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
    #     """
    #     #假如env_ids = tensor([0, 2, 4])，这表示第 0、2、4 个环境该重新采样命令了
    #     #定期挑出需要更新命令（command）的环境体编号，以便在训练中为这些环境赋予新的任务目标（比如速度、方向等），从而实现条件策略训练。
    #     #env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
    #     #print(self.cfg.commands.resampling_time / self.dt)
    #     #print((self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0))
    #     #self._resample_commands(env_ids)
    #     self.phase_length_buf += 1

    #     self._resample_commands()
    #     if self.cfg.commands.heading_command:
    #         forward = quat_apply(self.base_quat, self.forward_vec)
    #         heading = torch.atan2(forward[:, 1], forward[:, 0])
    #         self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
        
    #     if self.cfg.terrain.measure_heights:
    #         self.measured_heights = self._get_heights()

    #     if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
    #         self._push_robots()

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        self.phase_length_buf += 1
        #print(self.phase_length_buf)
        #print(self.episode_length_buf)episode_length_buf和phase_length_buf一样
        #print(self.common_step_counter)
        self._resample_commands()
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        # print(self.dt) # 0.01
        # print(self.cfg.domain_rand.update_step)
        # print(self.cfg.domain_rand.push_duration)
        # print(self.cfg.domain_rand.push_interval)
        if self.cfg.domain_rand.push_robots:
            i = int(self.common_step_counter/self.cfg.domain_rand.update_step)#每48000 step，就把推的持续时间增加一个档次。
            if i >= len(self.cfg.domain_rand.push_duration):
                i = len(self.cfg.domain_rand.push_duration) - 1
            duration = self.cfg.domain_rand.push_duration[i]/self.dt
            if self.common_step_counter % self.cfg.domain_rand.push_interval <= duration:
                self._push_robots()
            else:
                self.rand_push_force.zero_()
                self.rand_push_torque.zero_()

    # def _post_physics_step_callback(self):
    #     """ Callback called before computing terminations, rewards, and observations
    #         Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
    #     """
    #     # 
    #     env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
    #     self._resample_commands(env_ids)
    #     if self.cfg.commands.heading_command:
    #         forward = quat_apply(self.base_quat, self.forward_vec)
    #         heading = torch.atan2(forward[:, 1], forward[:, 0])
    #         self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

    #     if self.cfg.terrain.measure_heights:
    #         self.measured_heights = self._get_heights()

    #     if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
    #         self._push_robots()

    # def _resample_commands(self, env_ids):
    #     """ Randommly select commands of some environments

    #     Args:
    #         env_ids (List[int]): Environments ids for which new commands are needed
    #     """
    #     self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     # num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading
    #     if self.cfg.commands.heading_command:
    #         self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     else:
    #         self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

    #     # set small commands to zero
    #     self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :3], dim=1) > 0.2).unsqueeze(1)
    
    def _resample_commands(self):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        for i in range(len(self.cfg.commands.gait)):
            # if env finish current gait type, resample command for next gait
            env_ids = (self.episode_length_buf == self.gait_time[:,i]).nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                # according to gait type create a name
                name = '_resample_' + self.cfg.commands.gait[i] + '_command'
                # get function from self based on name
                resample_command = getattr(self, name)
                # resample_command stands for _resample_stand_command/_resample_walk_sagittal_command/...
                resample_command(env_ids)

    def _resample_stand_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)    

    def _resample_walk_omnidirectional_command(self,env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.05).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        p_gains = self.p_gains * self.joint_kp_coeffs
        d_gains = self.d_gains * self.joint_kd_coeffs

        torques = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        #unwrap_tensor 把 PyTorch 张量转为 原生 GPU/CPU 指针
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            # 在def _get_env_origins(self)，非平地为true
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            # 在 def _get_env_origins(self)，平地为false
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        
        # base velocities
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.05, 0.05, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        if self.cfg.asset.fix_base_link:
            self.root_states[env_ids, 7:13] = 0
            self.root_states[env_ids, 2] += 1.8
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
        
    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        #位置（x, y, z）旋转（四元数）线速度（vx, vy, vz）角速度（wx, wy, wz）
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        #每个关节的位置（角度或位移）每个关节的速度
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)
        #self.feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        #self.feet_euler_xyz = get_euler_xyz_tensor(self.feet_quat)

        #print(self.rigid_state.size())#torch.Size([4096, 13, 13])13个link   miao输出:torch.Size([2, 20, 13])
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        #print(get_axis_params(-1., self.up_axis_idx))#输出[0.0, 0.0, -1.0]
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_rigid_state = torch.zeros_like(self.rigid_state)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            # print(name)
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():

                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[:, i] = 0.
                self.d_gains[:, i] = 0.
                print(f"PD gain of joint {name} were not defined, setting them to zero")
        

        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_joint_pd_target = self.default_dof_pos.clone()
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(
                self.num_envs, self.cfg.env.single_num_privileged_obs, dtype=torch.float, device=self.device))
        
        self.gait_time = torch.zeros(self.num_envs, len(self.cfg.commands.gait) ,dtype=torch.int, device=self.device, requires_grad=False)
        self.phase_length_buf = torch.zeros(
           self.num_envs, device=self.device, dtype=torch.long)
        self.gait_start = torch.randint(0, 2, (self.num_envs,)).to(self.device)*0.5

        env = self.envs[0]
        actor_handle = self.gym.get_actor_handle(env, 0)
        
        # 获取刚体数量
        #Total rigid bodies: 20
        # RB 0: base_link
        # RB 1: loin_yaw_Link
        # RB 2: leg_l1_link
        # RB 3: leg_l2_link
        # RB 4: leg_l3_link
        # RB 5: leg_l4_link
        # RB 6: leg_l5_link
        # RB 7: leg_r1_link
        # RB 8: leg_r2_link
        # RB 9: leg_r3_link
        # RB 10: leg_r4_link
        # RB 11: leg_r5_link
        # RB 12: l_shoulder_pitch_link
        # RB 13: l_shoulder_roll_link
        # RB 14: l_shoulder_yaw_link
        # RB 15: l_arm_pitch_link
        # RB 16: r_shoulder_pitch_link
        # RB 17: r_shoulder_roll_link
        # RB 18: r_shoulder_yaw_link
        # RB 19: r_arm_pitch_link

        # Total rigid bodies: 13
        # RB 0: base_link
        # RB 1: left_leg_roll_link
        # RB 2: left_leg_yaw_link
        # RB 3: left_leg_pitch_link
        # RB 4: left_knee_link
        # RB 5: left_ankle_pitch_link
        # RB 6: left_ankle_roll_link
        # RB 7: right_leg_roll_link
        # RB 8: right_leg_yaw_link
        # RB 9: right_leg_pitch_link
        # RB 10: right_knee_link
        # RB 11: right_ankle_pitch_link
        # RB 12: right_ankle_roll_link
        rb_count = self.gym.get_actor_rigid_body_count(env, actor_handle)
        print(f"Total rigid bodies: {rb_count}")
        # print(len(self.envs))输出4096
        rb_names = []
        try:
            # 优先使用新版 API（一次性获取所有刚体名）
            if hasattr(self.gym, "get_actor_rigid_body_names"):
                rb_names = self.gym.get_actor_rigid_body_names(env, actor_handle)
                for i, name in enumerate(rb_names):
                    print(f"RB {i}: {name}")
            else:
                raise AttributeError("找不到可用的 Isaac Gym 刚体名称 API")
        except Exception as e:
            print(f"获取刚体名称时出错: {e}")


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))
        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        #print(len(self.episode_sums.keys()))输出22
        #print(self.episode_sums.keys())
    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def fix_dof_props_asset(self, props):
         # 从 MuJoCo 模型中读取 jnt_stiffness（关节刚度），转换成 float32 的 PyTorch 张量，并搬移到指定设备（比如 GPU 或 CPU）
        jnt_stiffness = torch.from_numpy(self.mujoco_model.jnt_stiffness).to(self.device).to(torch.float32)
        dof_frictionloss = torch.from_numpy(self.mujoco_model.dof_frictionloss).to(self.device).to(torch.float32)
        actuator_ctrlrange = torch.from_numpy(self.mujoco_model.actuator_ctrlrange).to(self.device)
        for i in range(len(props)):
             # 断言当前 DOF 必须有限位（hasLimits），否则报错
            #print(len(props))#输出19
            assert props["hasLimits"][i], "Joints must have limits!"
            props["stiffness"][i] = jnt_stiffness[i+1]
            props["friction"][i] = dof_frictionloss[i+6]
            assert abs(actuator_ctrlrange[i, 0]) == abs(actuator_ctrlrange[i, 1])
            props["effort"][i] = abs(actuator_ctrlrange[i, 0])
            props["velocity"][i] = 12
        return props

    def get_dof_axis(self):
        if self.asset_type == "mjcf":
            return torch.from_numpy(self.mujoco_model.jnt_axis[1:]).to(device=self.device)
        else:
            raise NotImplementedError

    def get_obs(self, name):
        if name == "dof_pos":
            return self.dof_pos + self.joint_pos_biases
        elif name in [
            "dof_vel", "actions",
            "base_lin_vel", "base_ang_vel",
            "rand_push_force", "rand_push_torque",
            "friction_coeffs", "restitution_coeffs", "base_mass_coeffs", "base_com_coeffs",
            "joint_friction_coeffs", "joint_armature_coeffs", "joint_pos_biases",
            "joint_kp_coeffs", "joint_kd_coeffs", "base_euler_bias"
        ]:
            assert len(getattr(self, name).shape) == 2, (
                f"Observation shape must be (num_envs, num_obs), but {name} is {getattr(self, name).shape}"
            )
            return getattr(self, name)
        elif name.startswith("base_euler"):
            axis = list(name.replace("base_euler_", ""))
            assert len(axis) == len(set(axis)) and set(axis).issubset({"x", "y", "z"})
            index = ["xyz".index(ax) for ax in axis]
            return self.base_euler_xyz[:, index] + self.base_euler_bias[:, index]
        elif name == "command_input":
            phase = self._get_phase() #cycle_time = self.cfg.rewards.cycle_time
                                      #phase = self.episode_length_buf * self.dt / cycle_time
            sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
            cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
            return torch.cat((sin_pos, cos_pos, self.commands[:, :3]), dim=1)
        elif name == "stance_mask":
            return self._get_gait_phase()
        elif name == "contact_mask":
            return self.contact_forces[:, self.feet_indices, 2] > 5.
        elif name == "target_dof_pos":
            self.compute_ref_state()
            return self.ref_dof_pos
        elif name == "measure_heights":
            return torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.)
        else:
            raise NotImplemented

    def generate_random(self, range, num=1):
        return torch_rand_float(range[0], range[1], (self.num_envs, num), device=self.device)

    def generate_constant(self, num, value):
        return torch.full((self.num_envs, num), value, dtype=torch.float32, device=self.device)

    def init_domain_randomization(self):
        dr = self.cfg.domain_rand

        if dr.randomize_friction:
            self.friction_coeffs = self.generate_random(dr.friction_range)
        else:
            self.friction_coeffs = self.generate_constant(1, 0)
        if dr.randomize_restitution:
            self.restitution_coeffs = self.generate_random(dr.restitution_range)
        else:
            self.restitution_coeffs = self.generate_constant(1, 0)
        
        
        if dr.randomize_base_mass:
            self.base_mass_coeffs = self.generate_random(dr.added_mass_range)
        else:
            self.base_mass_coeffs = self.generate_constant(1, 0)
        if dr.randomize_com_displacement:
            self.base_com_coeffs = self.generate_random(dr.com_displacement_range, num=3)
        else:
            self.base_com_coeffs = self.generate_constant(3, 0)
        # if dr.randomize_base_inertia:
        #     self.base_inertia_x = self.generate_random(dr.base_inertia_x_range)
        #     self.base_inertia_y = self.generate_random(dr.base_inertia_y_range)
        #     self.base_inertia_z = self.generate_random(dr.base_inertia_z_range)
        
        # if dr.randomize_link_mass:
        #     self.link_mass_coeffs = self.generate_random(dr.added_link_mass_range)

        # if dr.randomize_link_com_displacement:
        #     self.link_com_coeffs = self.generate_random(dr.link_com_displacement_range, num=3)

        # if dr.randomize_link_inertia:
        #     self.link_inertia_x = self.generate_random(dr.link_inertia_x_range)
        #     self.link_inertia_y = self.generate_random(dr.link_inertia_y_range)
        #     self.link_inertia_z = self.generate_random(dr.link_inertia_z_range)

        
        
        if dr.randomize_joint_friction:
            self.joint_friction_coeffs = self.generate_random(dr.joint_friction_range, num=self.num_dof)
        else:
            self.joint_friction_coeffs = self.generate_constant(self.num_dof, 1)   
        if dr.randomize_joint_armature:
            self.joint_armature_coeffs = self.generate_random(dr.joint_armature_range, num=self.num_dof)
        else:
            self.joint_armature_coeffs = self.generate_constant(self.num_dof, 1)
        if dr.randomize_joint_pos_bias:
            self.joint_pos_biases = self.generate_random(dr.joint_pos_bias_range, num=self.num_dof)
        else:
            self.joint_pos_biases = self.generate_constant(self.num_dof, 0)
        if dr.randomize_joint_kp:
            self.joint_kp_coeffs = self.generate_random(dr.joint_kp_range, num=self.num_dof)
        else:
            self.joint_kp_coeffs = self.generate_constant(self.num_dof, 1)
        if dr.randomize_joint_kd:
            self.joint_kd_coeffs = self.generate_random(dr.joint_kd_range, num=self.num_dof)
        else:
            self.joint_kd_coeffs = self.generate_constant(self.num_dof, 1)
        if dr.randomize_base_euler_bias:
            self.base_euler_bias = self.generate_random(dr.base_euler_bias_range, num=3)
        else:
            self.base_euler_bias = self.generate_constant(3, 0)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        xml_root = ET.parse(asset_path).getroot()
        if xml_root.tag == "mujoco":
            self.asset_type = "mjcf"
            import mujoco
            xml_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            self.asset_type = "urdf"
            self.mujoco_model = None
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        if self.asset_type == "mjcf":
            dof_props_asset = self.fix_dof_props_asset(dof_props_asset)
        #长度为20(miao_arm)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        #print(self.asset_type)输出mjcf
        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = self.cfg.asset.foot_names
        knee_names = self.cfg.asset.knee_names
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()#给self.env_origins[:, 2]赋值
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.init_domain_randomization()

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            # 在 pos 的前两个维度上，加入 [-1.0, 1.0] 范围内的随机扰动
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            #给每个link随机化摩擦、刚性
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            
            dof_props = self._process_dof_props(dof_props_asset, i)
            
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            #给base_link质量加一些随机化
            body_props = self._process_rigid_body_props(body_props, i)
            #print("Number of bodies:", len(body_props))fixed arm输出20

            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
        #print(len(feet_names))#输出2
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        #print(self.termination_contact_indices)#输出tensor([ 0, 12, 13, 14, 15, 16, 17, 18, 19], device='cuda:0')
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        #控制器动作更新的时间间隔(策略网络每隔多长时间输出新的动作)=控制器每次动作持续的仿真步数 × 每步仿真时间
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        #print(self.reward_scales.keys())
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        #print(self.max_episode_length)#输出2400
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
         # 遍历所有仿真环境（每个环境中有一个机器人）
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heightXBotL = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heightXBotL)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
