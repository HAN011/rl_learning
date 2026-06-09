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


from humanoid.envs import *
from humanoid.utils import get_args, task_registry, helpers

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    curriculum_stage = helpers.get_curriculum_stage(env_cfg)
    random_init_ep_len = bool(getattr(args, "random_init_ep_len", False))
    empirical_normalization = bool(train_cfg.runner.empirical_normalization)
    use_ref_actions = bool(getattr(env_cfg.env, "use_ref_actions", False))
    residual_action_scale = float(getattr(env_cfg.env, "residual_action_scale", 0.0))
    ref_cfg = {
        "ref_kp_pitch": float(getattr(env_cfg.env, "ref_kp_pitch", 0.0)),
        "ref_kd_pitch": float(getattr(env_cfg.env, "ref_kd_pitch", 0.0)),
        "ref_ki_pitch": float(getattr(env_cfg.env, "ref_ki_pitch", 0.0)),
        "ref_kp_pos": float(getattr(env_cfg.env, "ref_kp_pos", 0.0)),
        "ref_kd_pos": float(getattr(env_cfg.env, "ref_kd_pos", 0.0)),
        "ref_action_clip": float(getattr(env_cfg.env, "ref_action_clip", 0.0)),
        "ref_pitch_i_clip": float(getattr(env_cfg.env, "ref_pitch_i_clip", 0.0)),
    }
    if curriculum_stage is not None:
        print(f"Training with curriculum_stage={curriculum_stage}")
    if random_init_ep_len:
        print(
            "WARNING: random_init_ep_len biases episode-based training metrics "
            "(for example timeout_ratio and episode_length). "
            "Use humanoid/scripts/eval.py for checkpoint selection."
        )
    print(f"empirical_normalization={empirical_normalization}")
    print(f"use_ref_actions={use_ref_actions}, residual_action_scale={residual_action_scale}")
    if use_ref_actions:
        print(f"ref_controller={ref_cfg}")

    metadata_path = helpers.save_run_metadata(
        ppo_runner.log_dir,
        {
            "task": args.task,
            "curriculum_stage": curriculum_stage,
            "experiment_name": train_cfg.runner.experiment_name,
            "run_name": train_cfg.runner.run_name,
            "random_init_ep_len": random_init_ep_len,
            "empirical_normalization": empirical_normalization,
            "use_ref_actions": use_ref_actions,
            "residual_action_scale": residual_action_scale,
            **ref_cfg,
        },
    )
    if metadata_path is not None:
        print(f"Saved run metadata to: {metadata_path}")

    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=random_init_ep_len,
    )

if __name__ == '__main__':
    args = get_args()
    train(args)
