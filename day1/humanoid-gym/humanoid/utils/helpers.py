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

import datetime
import json
import os
import re
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

RUN_METADATA_FILENAME = "run_metadata.json"


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def _get_sorted_run_names(root):
    try:
        runs = [
            run
            for run in os.listdir(root)
            if os.path.isdir(os.path.join(root, run))
        ]
        if "exported" in runs:
            runs.remove("exported")
        if not runs:
            raise ValueError("No runs in this directory: " + root)

        def run_sort_key(run_name):
            # Expected prefix format: "Mar07_10-12-54_..."
            try:
                month = datetime.datetime.strptime(run_name[:3], "%b").month
                day = int(run_name[3:5])
                rest = run_name[6:]
                return (month, day, rest)
            except (ValueError, IndexError):
                # Put non-standard folder names first in lexical fallback space.
                return (0, 0, run_name)

        runs.sort(key=run_sort_key)
    except:
        raise ValueError("No runs in this directory: " + root)
    return runs


def resolve_run_dir(root, load_run=-1):
    runs = _get_sorted_run_names(root)
    last_run = os.path.join(root, runs[-1])
    load_run_is_latest = load_run in (-1, "-1", None)
    if load_run_is_latest:
        return last_run
    return os.path.join(root, load_run)


def get_checkpoint_iteration(checkpoint_path):
    checkpoint_name = os.path.basename(checkpoint_path)
    match = re.match(r"model_(\d+)\.pt$", checkpoint_name)
    if match is None:
        raise ValueError(f"Checkpoint path does not match expected pattern model_<iter>.pt: {checkpoint_path}")
    return int(match.group(1))


def list_checkpoint_paths(root, load_run=-1, checkpoint=None, start=None, end=None, interval=None):
    run_dir = resolve_run_dir(root, load_run=load_run)
    models = [
        os.path.join(run_dir, file_name)
        for file_name in os.listdir(run_dir)
        if re.match(r"model_(\d+)\.pt$", file_name)
    ]
    if not models:
        raise ValueError(f"No checkpoint files found in run directory: {run_dir}")

    models.sort(key=get_checkpoint_iteration)
    if checkpoint not in (-1, "-1", None):
        target_path = os.path.join(run_dir, f"model_{int(checkpoint)}.pt")
        if not os.path.isfile(target_path):
            raise ValueError(f"Checkpoint file not found: {target_path}")
        return [target_path]

    filtered = []
    for model_path in models:
        iteration = get_checkpoint_iteration(model_path)
        if start is not None and iteration < int(start):
            continue
        if end is not None and iteration > int(end):
            continue
        if interval not in (None, 0):
            if iteration % int(interval) != 0:
                continue
        filtered.append(model_path)

    if not filtered:
        raise ValueError(
            f"No checkpoints matched filters in run directory: {run_dir} "
            f"(start={start}, end={end}, interval={interval})"
        )
    return filtered


def get_load_path(root, load_run=-1, checkpoint=-1):
    load_run = resolve_run_dir(root, load_run=load_run)
    # CLI may pass "-1" as a string; treat it the same as integer sentinel -1.
    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file and file.endswith(".pt")]
        if not models:
            raise ValueError(f"No checkpoint files found in run directory: {load_run}")
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def get_curriculum_stage(env_cfg):
    env = getattr(env_cfg, "env", None)
    if env is None or not hasattr(env, "curriculum_stage"):
        return None
    return int(env.curriculum_stage)


def save_run_metadata(log_dir, metadata):
    if log_dir is None:
        return None
    os.makedirs(log_dir, exist_ok=True)
    metadata_path = os.path.join(log_dir, RUN_METADATA_FILENAME)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return metadata_path


def load_run_metadata(run_dir):
    if not run_dir:
        return None
    metadata_path = os.path.join(run_dir, RUN_METADATA_FILENAME)
    if not os.path.isfile(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def extract_curriculum_stage(text):
    if not text:
        return None
    match = re.search(r"(?i)stage[_-]?(\d+)", text)
    if match is None:
        return None
    return int(match.group(1))


def infer_curriculum_stage_from_run(root, load_run=-1, checkpoint=-1):
    try:
        load_path = get_load_path(root, load_run=load_run, checkpoint=checkpoint)
    except Exception:
        return None, None, None

    run_dir = os.path.dirname(load_path)
    metadata = load_run_metadata(run_dir)
    if isinstance(metadata, dict) and metadata.get("curriculum_stage") is not None:
        try:
            return int(metadata["curriculum_stage"]), "metadata", load_path
        except (TypeError, ValueError):
            pass

    for candidate in (run_dir, load_path):
        stage = extract_curriculum_stage(candidate)
        if stage is not None:
            return stage, "path", load_path

    return None, None, load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        if getattr(args, "curriculum_stage", None) is not None and hasattr(env_cfg.env, "curriculum_stage"):
            env_cfg.env.curriculum_stage = args.curriculum_stage
        if getattr(args, "use_ref_actions", False):
            env_cfg.env.use_ref_actions = True
        if getattr(args, "residual_action_scale", None) is not None:
            env_cfg.env.residual_action_scale = args.residual_action_scale
        for field in (
            "ref_kp_pitch",
            "ref_kd_pitch",
            "ref_ki_pitch",
            "ref_kp_pos",
            "ref_kd_pos",
            "ref_action_clip",
            "ref_pitch_i_clip",
        ):
            value = getattr(args, field, None)
            if value is not None and hasattr(env_cfg.env, field):
                setattr(env_cfg.env, field, value)
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if getattr(args, "num_steps_per_env", None) is not None:
            cfg_train.runner.num_steps_per_env = args.num_steps_per_env
        if getattr(args, "save_interval", None) is not None:
            cfg_train.runner.save_interval = args.save_interval
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint
        if getattr(args, "learning_rate", None) is not None:
            cfg_train.algorithm.learning_rate = args.learning_rate
        if getattr(args, "entropy_coef", None) is not None:
            cfg_train.algorithm.entropy_coef = args.entropy_coef
        if getattr(args, "schedule", None) is not None:
            cfg_train.algorithm.schedule = args.schedule
        if getattr(args, "desired_kl", None) is not None:
            cfg_train.algorithm.desired_kl = args.desired_kl
        if getattr(args, "gamma", None) is not None:
            cfg_train.algorithm.gamma = args.gamma
        if getattr(args, "init_noise_std", None) is not None:
            cfg_train.policy.init_noise_std = args.init_noise_std
        if getattr(args, "no_empirical_normalization", False):
            cfg_train.runner.empirical_normalization = False

    return env_cfg, cfg_train


def get_args(extra_parameters=None):
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "two_wheel_balancer_ppo",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--resume",
            "action": "store_true",
            "default": False,
            "help": "Resume training from a checkpoint",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--run_name",
            "type": str,
            "help": "Name of the run. Overrides config file if provided.",
        },
        {
            "name": "--load_run",
            "type": str,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "Force display off at all times",
        },
        {
            "name": "--random_init_ep_len",
            "action": "store_true",
            "default": False,
            "help": "Randomize initial episode lengths before PPO rollouts.",
        },
        {
            "name": "--no_empirical_normalization",
            "action": "store_true",
            "default": False,
            "help": "Disable empirical observation normalization during training and inference.",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--num_envs",
            "type": int,
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--seed",
            "type": int,
            "help": "Random seed. Overrides config file if provided.",
        },
        {
            "name": "--max_iterations",
            "type": int,
            "help": "Maximum number of training iterations. Overrides config file if provided.",
        },
        {
            "name": "--num_steps_per_env",
            "type": int,
            "help": "Rollout steps per environment for each PPO iteration. Overrides config file if provided.",
        },
        {
            "name": "--save_interval",
            "type": int,
            "help": "Checkpoint save interval in PPO iterations. Overrides config file if provided.",
        },
        {
            "name": "--curriculum_stage",
            "type": int,
            "help": "Curriculum stage for env randomization. Overrides config file if provided.",
        },
        {
            "name": "--use_ref_actions",
            "action": "store_true",
            "default": False,
            "help": "Enable reference-controller actions and learn residual actions on top.",
        },
        {
            "name": "--residual_action_scale",
            "type": float,
            "help": "Residual action scale when --use_ref_actions is enabled. Overrides config file if provided.",
        },
        {
            "name": "--ref_kp_pitch",
            "type": float,
            "help": "Reference-controller proportional gain on pitch. Overrides config file if provided.",
        },
        {
            "name": "--ref_kd_pitch",
            "type": float,
            "help": "Reference-controller derivative gain on pitch rate. Overrides config file if provided.",
        },
        {
            "name": "--ref_ki_pitch",
            "type": float,
            "help": "Reference-controller integral gain on pitch. Overrides config file if provided.",
        },
        {
            "name": "--ref_kp_pos",
            "type": float,
            "help": "Reference-controller proportional gain on x position. Overrides config file if provided.",
        },
        {
            "name": "--ref_kd_pos",
            "type": float,
            "help": "Reference-controller derivative gain on x velocity. Overrides config file if provided.",
        },
        {
            "name": "--ref_action_clip",
            "type": float,
            "help": "Reference-controller action clip before residual combination. Overrides config file if provided.",
        },
        {
            "name": "--ref_pitch_i_clip",
            "type": float,
            "help": "Reference-controller pitch integral clip. Overrides config file if provided.",
        },
        {
            "name": "--learning_rate",
            "type": float,
            "help": "PPO learning rate. Overrides config file if provided.",
        },
        {
            "name": "--entropy_coef",
            "type": float,
            "help": "PPO entropy coefficient. Overrides config file if provided.",
        },
        {
            "name": "--schedule",
            "type": str,
            "choices": ["adaptive", "fixed"],
            "help": "PPO learning-rate schedule. Overrides config file if provided.",
        },
        {
            "name": "--desired_kl",
            "type": float,
            "help": "Target KL for adaptive PPO schedule. Overrides config file if provided.",
        },
        {
            "name": "--gamma",
            "type": float,
            "help": "PPO discount factor gamma. Overrides config file if provided.",
        },
        {
            "name": "--init_noise_std",
            "type": float,
            "help": "Initial policy action std for PPO exploration. Overrides config file if provided.",
        },
    ]
    if extra_parameters:
        custom_parameters.extend(extra_parameters)
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy", custom_parameters=custom_parameters
    )
    
    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def export_policy_as_jit(policy, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy_1.pt")
    model = copy.deepcopy(policy).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)


def export_policy_to_onnx(policy, path):
    import os
    import copy
    import torch

    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, "policy_onnx.onnx")
    model = copy.deepcopy(policy).to("cpu")
    model.eval()  # 设置模型为评估模式

    # Infer input dimension from current policy to avoid stale hard-coded obs size.
    batch_size = 1

    num_observations = None
    if hasattr(model, "_mean"):
        num_observations = int(model._mean.shape[-1])
    elif isinstance(model, torch.nn.Sequential):
        for sub_module in model:
            if hasattr(sub_module, "_mean"):
                num_observations = int(sub_module._mean.shape[-1])
                break
            if isinstance(sub_module, torch.nn.Linear):
                num_observations = int(sub_module.in_features)
                break
    elif isinstance(model, torch.nn.Linear):
        num_observations = int(model.in_features)

    if num_observations is None:
        raise ValueError("Could not infer observation dimension for ONNX export.")

    dummy_input = torch.randn(batch_size, num_observations)

    # 导出模型为 ONNX 格式
    torch.onnx.export(
        model,                      # 模型
        dummy_input,                # 示例输入
        model_path,                 # 导出路径
        export_params=True,         # 导出模型参数
        opset_version=11,           # ONNX opset 版本，可以根据需要调整
        do_constant_folding=True,   # 进行常量折叠优化
        input_names=['input'],        # 输入节点名称，可以根据需要调整
        output_names=['output'],   # 输出节点名称，可以根据需要调整
    )
    print(f"模型已导出为 ONNX 格式，保存路径：{model_path}")
