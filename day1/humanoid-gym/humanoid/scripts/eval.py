# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import json
import os
import time

import numpy as np

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *
from humanoid.utils import get_args, helpers, task_registry
import torch


DEFAULT_EVAL_EPISODES = 1024
DEFAULT_STAGE_GATE_TIMEOUT_RATIO = 0.95


def _get_eval_args():
    extra_parameters = [
        {
            "name": "--eval_episodes",
            "type": int,
            "default": DEFAULT_EVAL_EPISODES,
            "help": "Minimum number of complete episodes to collect for each checkpoint.",
        },
        {
            "name": "--checkpoint_interval",
            "type": int,
            "default": 0,
            "help": "Only evaluate checkpoints where iter %% interval == 0. Use 0 to evaluate all checkpoints.",
        },
        {
            "name": "--checkpoint_start",
            "type": int,
            "help": "Minimum checkpoint iteration to evaluate.",
        },
        {
            "name": "--checkpoint_end",
            "type": int,
            "help": "Maximum checkpoint iteration to evaluate.",
        },
        {
            "name": "--eval_output",
            "type": str,
            "help": "Optional JSON output path. Defaults to a file inside the evaluated run directory.",
        },
        {
            "name": "--stage_gate_timeout_ratio",
            "type": float,
            "default": DEFAULT_STAGE_GATE_TIMEOUT_RATIO,
            "help": "Minimum true timeout ratio required to declare the checkpoint ready for the next curriculum stage.",
        },
        {
            "name": "--stage_gate_target",
            "type": int,
            "default": 1,
            "help": "Target curriculum stage after passing evaluation. Used for reporting only.",
        },
        {
            "name": "--stochastic_eval",
            "action": "store_true",
            "default": False,
            "help": "Sample actions from the policy distribution instead of using deterministic mean actions.",
        },
    ]
    return get_args(extra_parameters=extra_parameters)


def _resolve_log_root(args, train_cfg):
    experiment_name = args.experiment_name or train_cfg.runner.experiment_name
    return os.path.join(LEGGED_GYM_ROOT_DIR, "logs", experiment_name)


def _resolve_curriculum_stage(args, env_cfg, train_cfg, checkpoint_path):
    if not hasattr(env_cfg.env, "curriculum_stage"):
        return None, "unsupported"
    if args.curriculum_stage is not None:
        return int(args.curriculum_stage), "cli"

    log_root = _resolve_log_root(args, train_cfg)
    run_dir = os.path.dirname(checkpoint_path)
    run_name = os.path.basename(run_dir)
    checkpoint_iter = helpers.get_checkpoint_iteration(checkpoint_path)
    stage, source, _ = helpers.infer_curriculum_stage_from_run(
        log_root,
        load_run=run_name,
        checkpoint=checkpoint_iter,
    )
    if stage is None:
        return int(env_cfg.env.curriculum_stage), "config_default"
    return int(stage), source


def _resolve_eval_output(args, run_dir, current_stage):
    if args.eval_output:
        return args.eval_output
    target_stage = int(args.stage_gate_target)
    return os.path.join(run_dir, f"eval_stage{current_stage}_to_stage{target_stage}.json")


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _reason_count(infos, key, num_resets):
    episode_info = infos.get("episode", {})
    metric_name = f"reset_{key}_ratio"
    if metric_name not in episode_info:
        return 0.0
    return _to_float(episode_info[metric_name]) * float(num_resets)


def _build_runner(args, checkpoint_path):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    stage, stage_source = _resolve_curriculum_stage(args, env_cfg, train_cfg, checkpoint_path)
    if stage is not None:
        env_cfg.env.curriculum_stage = stage

    args.headless = True
    args.resume = False
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=None,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    return env, runner, train_cfg, stage, stage_source


def _reset_env_from_reset_idx(env):
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(env_ids)
    env.compute_observations()
    obs = env.get_observations()
    privileged_obs = env.get_privileged_observations()
    return obs, privileged_obs


def _evaluate_checkpoint(env, runner, checkpoint_path, eval_episodes, stochastic_eval=False):
    runner.load(checkpoint_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    actor_critic = runner.alg.actor_critic
    obs, _ = _reset_env_from_reset_idx(env)
    obs = obs.to(env.device)

    episode_lengths = []
    completed_episodes = 0
    timeout_count = 0
    base_contact_count = 0.0
    over_tilt_count = 0.0
    over_tilt_raw_count = 0.0
    low_base_count = 0.0
    active_episode_lengths = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    rollout_steps = 0
    start_time = time.time()

    with torch.no_grad():
        while completed_episodes < eval_episodes:
            if stochastic_eval:
                norm_obs = runner.obs_normalizer(obs) if runner.cfg["empirical_normalization"] else obs
                actions = actor_critic.act(norm_obs)
            else:
                actions = policy(obs.detach())
            obs, _, _, dones, infos = env.step(actions.detach())
            obs = obs.to(env.device)
            active_episode_lengths += 1
            rollout_steps += 1

            done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
            if len(done_ids) == 0:
                continue

            lengths = active_episode_lengths[done_ids].cpu().numpy().tolist()
            episode_lengths.extend(lengths)
            completed_episodes += len(lengths)

            if "time_outs" in infos:
                timeout_count += int(infos["time_outs"][done_ids].sum().item())
            else:
                timeout_count += int(round(_reason_count(infos, "timeout", len(done_ids))))

            base_contact_count += _reason_count(infos, "base_contact", len(done_ids))
            over_tilt_count += _reason_count(infos, "over_tilt", len(done_ids))
            over_tilt_raw_count += _reason_count(infos, "over_tilt_raw", len(done_ids))
            low_base_count += _reason_count(infos, "low_base", len(done_ids))

            active_episode_lengths[done_ids] = 0

    elapsed = time.time() - start_time
    episode_lengths_np = np.asarray(episode_lengths, dtype=np.float64)
    total_episodes = int(episode_lengths_np.size)
    timeout_ratio = float(timeout_count / total_episodes) if total_episodes > 0 else 0.0

    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_iteration": helpers.get_checkpoint_iteration(checkpoint_path),
        "stochastic_eval": bool(stochastic_eval),
        "episodes_requested": int(eval_episodes),
        "episodes_completed": total_episodes,
        "timeout_count": int(timeout_count),
        "timeout_ratio": timeout_ratio,
        "failure_ratio": float(1.0 - timeout_ratio),
        "mean_episode_length_steps": float(episode_lengths_np.mean()) if total_episodes > 0 else 0.0,
        "median_episode_length_steps": float(np.median(episode_lengths_np)) if total_episodes > 0 else 0.0,
        "p10_episode_length_steps": float(np.percentile(episode_lengths_np, 10)) if total_episodes > 0 else 0.0,
        "p90_episode_length_steps": float(np.percentile(episode_lengths_np, 90)) if total_episodes > 0 else 0.0,
        "mean_episode_length_s": float(episode_lengths_np.mean() * env.dt) if total_episodes > 0 else 0.0,
        "base_contact_ratio": float(base_contact_count / total_episodes) if total_episodes > 0 else 0.0,
        "over_tilt_ratio": float(over_tilt_count / total_episodes) if total_episodes > 0 else 0.0,
        "over_tilt_raw_ratio": float(over_tilt_raw_count / total_episodes) if total_episodes > 0 else 0.0,
        "low_base_ratio": float(low_base_count / total_episodes) if total_episodes > 0 else 0.0,
        "rollout_policy_steps": int(rollout_steps),
        "rollout_wall_time_s": float(elapsed),
        "rollout_policy_fps": float((rollout_steps * env.num_envs) / elapsed) if elapsed > 0 else 0.0,
    }


def _select_best_result(results):
    return max(
        results,
        key=lambda result: (
            result["timeout_ratio"],
            result["mean_episode_length_steps"],
            result["checkpoint_iteration"],
        ),
    )


def _print_result(result):
    print(
        "[eval] "
        f"checkpoint={result['checkpoint_iteration']} "
        f"episodes={result['episodes_completed']} "
        f"timeout_ratio={result['timeout_ratio']:.4f} "
        f"failure_ratio={result['failure_ratio']:.4f} "
        f"mean_len_s={result['mean_episode_length_s']:.2f} "
        f"over_tilt_ratio={result['over_tilt_ratio']:.4f} "
        f"low_base_ratio={result['low_base_ratio']:.4f}"
    )


def main(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    log_root = _resolve_log_root(args, train_cfg)
    checkpoint_paths = helpers.list_checkpoint_paths(
        log_root,
        load_run=args.load_run,
        checkpoint=args.checkpoint,
        start=args.checkpoint_start,
        end=args.checkpoint_end,
        interval=args.checkpoint_interval,
    )
    run_dir = os.path.dirname(checkpoint_paths[0])

    env, runner, train_cfg, stage, stage_source = _build_runner(args, checkpoint_paths[0])
    print(f"Evaluating run: {run_dir}")
    print(f"Checkpoint count: {len(checkpoint_paths)}")
    print(f"Curriculum stage: {stage} ({stage_source})")
    print(f"Eval episodes per checkpoint: {args.eval_episodes}")
    print(f"Stage gate timeout ratio: {args.stage_gate_timeout_ratio:.4f}")
    print(f"Action mode: {'stochastic' if args.stochastic_eval else 'deterministic'}")

    results = []
    for checkpoint_path in checkpoint_paths:
        result = _evaluate_checkpoint(
            env,
            runner,
            checkpoint_path,
            eval_episodes=args.eval_episodes,
            stochastic_eval=bool(args.stochastic_eval),
        )
        result["curriculum_stage"] = int(stage) if stage is not None else None
        result["curriculum_stage_source"] = stage_source
        results.append(result)
        _print_result(result)

    best_result = _select_best_result(results)
    ready_for_next_stage = best_result["timeout_ratio"] >= float(args.stage_gate_timeout_ratio)
    output_path = _resolve_eval_output(args, run_dir, current_stage=int(stage) if stage is not None else -1)

    summary = {
        "task": args.task,
        "experiment_name": args.experiment_name or train_cfg.runner.experiment_name,
        "run_dir": run_dir,
        "curriculum_stage": int(stage) if stage is not None else None,
        "curriculum_stage_source": stage_source,
        "target_stage": int(args.stage_gate_target),
        "stage_gate_timeout_ratio": float(args.stage_gate_timeout_ratio),
        "stochastic_eval": bool(args.stochastic_eval),
        "ready_for_target_stage": ready_for_next_stage,
        "best_checkpoint": best_result,
        "results": results,
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(
        "[best] "
        f"checkpoint={best_result['checkpoint_iteration']} "
        f"timeout_ratio={best_result['timeout_ratio']:.4f} "
        f"mean_len_s={best_result['mean_episode_length_s']:.2f} "
        f"ready_for_stage_{args.stage_gate_target}={ready_for_next_stage}"
    )
    print(f"Saved eval summary to: {output_path}")


if __name__ == "__main__":
    main(_get_eval_args())
