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


import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import MiaoArmHalfCfg
import torch
import time
from humanoid.scripts.KeyboardController import KeyboardController
import matplotlib.pyplot as plt

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

    @classmethod
    def update_from_controller(cls, stick_values):
        cls.vx = stick_values["left_stick_y"] * 0.6  # 前后移动
        cls.vy = stick_values["left_stick_x"] * 0.6  # 左右移动
        cls.dyaw = stick_values["right_stick_x"] * 0.5  # 转向

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    yaw_z=yaw_z
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    # quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    quat = data.qpos[[4, 5, 6, 3]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    # omega = data.sensor('angular-velocity').data.astype(np.double)
    omega = data.qvel[3:6].astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)#它根据当前 data 的状态推进一步，然后更新力学状态（关节位置、速度、接触力等）。
    #viewer = mujoco_viewer.MujocoViewer(model, data)#创建一个可视化窗口，用于实时渲染和查看模型运动状态。
    viewer = mujoco_viewer.MujocoViewer(model, data)

    # 设置摄像头跟踪 base_link
    base_id = model.body(name='base_link').id
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = base_id
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -15
    viewer.cam.azimuth = 90

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    actions_log = []   # 用来存储所有 action
    q_log = []   # 用来存储所有 action
    dq_log = []   # 用来存储所有 action
    omega_log = []   # 用来存储所有 action
    eu_ang_log = []   # 用来存储所有 action
    times_log = []     # 存储时间戳
    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0
    count_lowlevel2 = 0
    default_angle =np.zeros((cfg.env.num_actions),dtype=np.double)
    default_angle[0]=cfg.init_state.default_joint_angles['leg_l1_joint']
    default_angle[1]=cfg.init_state.default_joint_angles['leg_l2_joint']
    default_angle[2]=cfg.init_state.default_joint_angles['leg_l3_joint']
    default_angle[3]=cfg.init_state.default_joint_angles['leg_l4_joint']
    default_angle[4]=cfg.init_state.default_joint_angles['leg_l5_joint']
    
    default_angle[5]=cfg.init_state.default_joint_angles['leg_r1_joint']
    default_angle[6]=cfg.init_state.default_joint_angles['leg_r2_joint']
    default_angle[7]=cfg.init_state.default_joint_angles['leg_r3_joint']
    default_angle[8]=cfg.init_state.default_joint_angles['leg_r4_joint']
    default_angle[9]=cfg.init_state.default_joint_angles['leg_r5_joint']
    
    default_angle[10]=cfg.init_state.default_joint_angles['l_shoulder_pitch_joint']
    default_angle[11]=cfg.init_state.default_joint_angles['r_shoulder_pitch_joint']
    print("num_observations:",cfg.env.num_observations)#15*68=1020  15*44=660
    print("num_single_obs:",cfg.env.num_single_obs)#68  44
    start_time = time.time()
    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        
        # 更新cmd类中的值
        cmd.update_from_controller(keyboard_controller.get_stick_values())

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]#从 q 的尾部提取最后 num_actions 个元素
        dq = dq[-cfg.env.num_actions:]

        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            vel_norm = np.sqrt(cmd.vx**2 + cmd.vy**2 + cmd.dyaw**2)
            if  vel_norm <= cfg.commands.stand_com_threshold:
                    count_lowlevel = 0
                    #print(count_lowlevel)
            #start = time.time()
            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            #print(eu_ang)
            #在train.py里是self.episode_length_buf * self.dt / cycle_time
            obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)
            obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)
            #obs[0, 0] =0.0
            #obs[0, 1] =0.0
            #print(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)
            obs[0, 2] = cmd.vx
            obs[0, 3] = cmd.vy
            obs[0, 4] = cmd.dyaw
            obs[0, 5:5+12] = q
            obs[0, 5+12:5+12*2] = dq
            obs[0, 5+12*2:5+12*3] = action
            obs[0, 5+12*3:5+12*3+3] = omega
            obs[0, 5+12*3+3:5+12*3+6] = eu_ang
            # obs[0, 5:5+10] = default_angle
            # obs[0, 5+10:5+10*2] = 0
            # obs[0, 5+10*2:5+10*3] = 0
            # obs[0, 5+10*3:5+10*3+3] = 0
            # obs[0, 5+10*3+3:5+10*3+6] = 0

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
            
            #print(obs.shape)#miao:(1, 68)
            hist_obs.append(obs)
            hist_obs.popleft()
            #cfg.env.num_observations 通常等于 frame_stack × num_single_obs
            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            
            #start_time = time.time()
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()#这一步大概耗时0.8ms
            
            #end_time = time.time()
            #print((end_time - start_time))
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            #end = time.time()

            #print(f"update_action() 耗时: {(end - start) * 1000:.3f} ms")
            actions_log.append(action.copy()) 
            q_log.append(q.copy())
            dq_log.append(dq.copy())
            omega_log.append(omega.copy())
            eu_ang_log.append(eu_ang.copy()) 
             
            times_log.append(time.time() - start_time)
            target_q = action * cfg.control.action_scale+default_angle
            
            viewer.render()
            
        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        
        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds)  # Calc torques
        
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau

        mujoco.mj_step(model, data)
        #end_time = time.time()
        #print((end_time - start_time))
        #start_time = time.time()
        count_lowlevel += 1

    actions_log = np.array(actions_log)   # shape: (steps, 10)
    q_log = np.array(q_log)   # shape: (steps, 10)
    dq_log = np.array(dq_log)   # shape: (steps, 10)
    omega_log = np.array(omega_log)   # shape: (steps, 10)
    eu_ang_log = np.array(eu_ang_log)   # shape: (steps, 10)
    
    times_log = np.array(times_log)
    fig, axes = plt.subplots(cfg.env.num_actions, 1, figsize=(20, 20), sharex=True)

    for i in range(actions_log.shape[1]):  # 遍历每个 action
        axes[i].plot(times_log, actions_log[:, i])
        axes[i].set_ylabel(f"Action {i}")
        axes[i].grid(True)
    axes[-1].set_xlabel("Time [s]")  # 只在最后一个子图加横坐标
    plt.suptitle("Actions over time")
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 防止标题和子图重叠
    plt.show()

    # for i in range(q_log.shape[1]):  # 遍历每个 action
    #     axes[i].plot(times_log, q_log[:, i])
    #     axes[i].set_ylabel(f"q {i}")
    #     axes[i].grid(True)
    # axes[-1].set_xlabel("Time [s]")  # 只在最后一个子图加横坐标
    # plt.suptitle("q over time")
    # plt.tight_layout(rect=[0, 0, 1, 0.97])  # 防止标题和子图重叠
    # plt.show()

    # for i in range(dq_log.shape[1]):  # 遍历每个 action
    #     axes[i].plot(times_log, dq_log[:, i])
    #     axes[i].set_ylabel(f"dq {i}")
    #     axes[i].grid(True)
    # axes[-1].set_xlabel("Time [s]")  # 只在最后一个子图加横坐标
    # plt.suptitle("dq over time")
    # plt.tight_layout(rect=[0, 0, 1, 0.97])  # 防止标题和子图重叠
    # plt.show()

    # for i in range(omega_log.shape[1]):  # 遍历每个 action
    #     axes[i].plot(times_log, omega_log[:, i])
    #     axes[i].set_ylabel(f"omega {i}")
    #     axes[i].grid(True)
    # axes[-1].set_xlabel("Time [s]")  # 只在最后一个子图加横坐标
    # plt.suptitle("omega over time")
    # plt.tight_layout(rect=[0, 0, 1, 0.97])  # 防止标题和子图重叠
    # plt.show()

    # for i in range(eu_ang_log.shape[1]):  # 遍历每个 action
    #     axes[i].plot(times_log, eu_ang_log[:, i])
    #     axes[i].set_ylabel(f"eu_ang {i}")
    #     axes[i].grid(True)
    # axes[-1].set_xlabel("Time [s]")  # 只在最后一个子图加横坐标
    # plt.suptitle("eu_ang over time")
    # plt.tight_layout(rect=[0, 0, 1, 0.97])  # 防止标题和子图重叠
    # plt.show()

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    keyboard_controller = KeyboardController()   #init keyboard controller
    
    class Sim2simCfg(MiaoArmHalfCfg):

        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/miao_arm/mjcf/robot_half_arm.xml'
            sim_duration = 10.0
            dt = 0.001
            decimation = 10

        class robot_config:
            # kps = np.array([30.] * 10, dtype=np.double)
            # kds = np.array([3.] * 10, dtype=np.double)
            kps = np.array([30,30,30,30,30,   30,30,30,30,30,     10,10], dtype=np.double)
            kds = np.array([3,3,3,3,3,      3,3,3,3,3,             0.7,0.7] , dtype=np.double)

            tau_limit = np.array([28,28,28,28,28,      28,28,28,28,28,        7,7] , dtype=np.double)
    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
