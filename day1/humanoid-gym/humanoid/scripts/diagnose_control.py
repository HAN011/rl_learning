# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

from humanoid.envs import *
from humanoid.utils import get_args, task_registry

import torch


def _get_diag_args():
    extra_parameters = [
        {
            "name": "--steps_per_segment",
            "type": int,
            "default": 20,
            "help": "Number of simulation steps for each torque segment.",
        },
        {
            "name": "--torque",
            "type": float,
            "default": 5.0,
            "help": "Fixed wheel torque in N*m for the positive/negative segments.",
        },
    ]
    return get_args(extra_parameters=extra_parameters)


def _build_env(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    if hasattr(env_cfg.env, "curriculum_stage") and args.curriculum_stage is not None:
        env_cfg.env.curriculum_stage = args.curriculum_stage

    # Keep the diagnostic deterministic and avoid resets masking the response.
    env_cfg.env.num_envs = 1
    env_cfg.env.reset_angle_range = 0.0
    env_cfg.env.reset_wheel_vel = 0.0
    env_cfg.env.reset_base_lin_vel = 0.0
    env_cfg.env.reset_base_ang_vel = 0.0
    env_cfg.env.use_ref_actions = False
    env_cfg.env.max_tilt_deg = 179.0
    env_cfg.env.min_base_height = -10.0
    env_cfg.env.base_contact_force_threshold = 1e9
    env_cfg.env.episode_length_s = max(float(env_cfg.env.episode_length_s), 10.0)

    env_cfg.noise.add_noise = False
    env_cfg.noise.noise_level = 0.0
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.action_delay = 0.0
    env_cfg.domain_rand.action_noise = 0.0

    args.headless = True
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    return env, env_cfg


def _reset_env(env):
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(env_ids)
    env.compute_observations()


def _action_from_torque(env, torque_nm):
    action_value = float(torque_nm) / float(env.cfg.control.action_scale)
    clip_value = float(env.cfg.normalization.clip_actions)
    action_value = max(min(action_value, clip_value), -clip_value)
    return torch.full((env.num_envs, env.num_actions), action_value, device=env.device)


def _sign_relation(left_vel, right_vel, eps=1e-6):
    if abs(left_vel) <= eps or abs(right_vel) <= eps:
        return "zero"
    if left_vel * right_vel > 0.0:
        return "same"
    return "opposite"


def _run_segment(env, label, torque_nm, num_steps):
    _reset_env(env)
    actions = _action_from_torque(env, torque_nm)
    command_torque = float(actions[0, 0].item()) * float(env.cfg.control.action_scale)

    print("")
    print(f"[segment] {label}: torque_cmd={command_torque:.4f} N*m, action={actions[0, 0].item():.4f}")
    print("step pitch wheel_L_vel wheel_R_vel x_pos torque_L torque_R vel_sign")

    same_count = 0
    opposite_count = 0
    zero_count = 0

    for step_idx in range(num_steps):
        env.step(actions)

        pitch = float(env.base_euler_xyz[0, 1].item())
        wheel_l_vel = float(env.dof_vel[0, 0].item())
        wheel_r_vel = float(env.dof_vel[0, 1].item())
        x_pos = float((env.root_states[0, 0] - env.env_origins[0, 0]).item())
        torque_l = float(env.torques[0, 0].item())
        torque_r = float(env.torques[0, 1].item())
        vel_sign = _sign_relation(wheel_l_vel, wheel_r_vel)

        if vel_sign == "same":
            same_count += 1
        elif vel_sign == "opposite":
            opposite_count += 1
        else:
            zero_count += 1

        print(
            f"{step_idx:02d} "
            f"{pitch:+.6f} "
            f"{wheel_l_vel:+.6f} "
            f"{wheel_r_vel:+.6f} "
            f"{x_pos:+.6f} "
            f"{torque_l:+.6f} "
            f"{torque_r:+.6f} "
            f"{vel_sign}"
        )

    print(
        f"[summary] {label}: same_sign_steps={same_count}, "
        f"opposite_sign_steps={opposite_count}, zero_sign_steps={zero_count}"
    )


def main(args):
    env, env_cfg = _build_env(args)
    print(f"task={args.task}")
    if hasattr(env_cfg.env, "curriculum_stage"):
        print(f"curriculum_stage={env_cfg.env.curriculum_stage}")
    print(f"action_scale={env.cfg.control.action_scale}")
    print(f"torque_limit={env.torque_limits.tolist()}")
    print(f"steps_per_segment={args.steps_per_segment}")
    print(f"fixed_torque={args.torque}")

    _run_segment(env, "zero", 0.0, args.steps_per_segment)
    _run_segment(env, "positive", abs(args.torque), args.steps_per_segment)
    _run_segment(env, "negative", -abs(args.torque), args.steps_per_segment)


if __name__ == "__main__":
    main(_get_diag_args())
