# TAL Branch Handoff

## Goal

Build the `tal_v1` branch on top of the shared reward framework already added to the codebase, while the main branch focuses on `cth_v1`.

Primary objective:

- improve high-speed stability
- reduce `out_of_track`
- preserve enough pace to remain competitive on lap time

Secondary objective:

- keep the TAL implementation modular so it can later serve as a training aid, not necessarily the final deployed reward

## Current Code Status

Shared interfaces are already prepared in:

- [DDPG_Torcs_PyTorch/gym_torcs.py](/home/ccczorange/RL-learning/ex3/DDPG_Torcs_PyTorch/gym_torcs.py)
- [DDPG_Torcs_PyTorch/test.py](/home/ccczorange/RL-learning/ex3/DDPG_Torcs_PyTorch/test.py)

What is already in place:

- `reward_mode.startswith("tal")` dispatch exists
- `compute_tal_reward(...)` exists
- `teacher_action` plumbing exists through env step and `info`
- `HeuristicTeacherPolicy` exists as a baseline teacher
- `tal_v1` profile exists in `test.py`
- checkpoint ranking now prioritizes valid lap completion, then collisions/out-of-track, then lap time

## Shared TAL Interface

### Reward entry

`compute_reward_and_termination(obs, obs_pre, applied_action, teacher_action=None)`

For TAL modes it routes to:

`compute_tal_reward(obs, obs_pre, applied_action, teacher_action)`

### Teacher hook

Teacher is created by:

`build_teacher_policy()`

Controlled by:

- `TORCS_TAL_TEACHER=heuristic`
- `TORCS_TAL_TEACHER=none`

Teacher output format:

```python
{
    "steer": float,
    "accel": float,
    "brake": float,
    "gear": int,
}
```

Teacher signal is attached to `info` as:

- `teacher_steer`
- `teacher_accel`
- `teacher_brake`

## What TAL Currently Does

Current `tal_v1` is a minimal usable scaffold, not the final intended TAL:

- alignment reward between `applied_action` and `teacher_action`
- small progress shaping using `distRaced`
- step penalty
- damage penalty
- collision termination
- out-of-track termination
- lap bonus

This is enough to run ablations, but not enough to claim paper-faithful trajectory-aided learning yet.

## Required Next Work

### 1. Replace heuristic teacher with a stronger teacher

Priority order:

1. improve heuristic teacher with speed scheduling based on local track geometry
2. add a waypoint / racing-line teacher
3. optionally add a classical controller wrapper that outputs steer and speed targets

The current heuristic teacher is fast to use but will cap TAL performance if left unchanged.

### 2. Make TAL reward closer to literature intent

Recommended direction:

- retain action-alignment term
- add trajectory-relative state term, not only action imitation
- use progress or forward-speed shaping only as auxiliary term

Suggested structure:

`r = w_align * r_align + w_progress * r_progress + w_state * r_state + r_terminal`

Where:

- `r_align`: match teacher steer / accel / brake
- `r_progress`: forward progress or `distRaced` increment
- `r_state`: penalty for track deviation and heading deviation
- `r_terminal`: hard collision/off-track/lap events

### 3. Evaluate whether TAL should keep the safety shell

Current default:

- `tal_*` uses the shared safety shell

Need explicit ablation:

1. `tal_v1` with safety shell on
2. `tal_v1` with safety shell off using `TORCS_ENABLE_SAFETY_SHELL=0`

This matters because:

- shell on improves training stability
- shell off better reveals whether the teacher signal itself is sufficient

## Recommended Experiment Matrix

### Round 1: teacher quality

Run:

1. `tal_v1` with current heuristic teacher
2. `tal_v1` with tuned heuristic teacher
3. `tal_v1` with shell off

Compare:

- `collision_count`
- `out_of_track_count`
- `avg_speed`
- `avg_dist_raced`
- first valid `best_lap_time`

### Round 2: reward decomposition

Try:

1. stronger alignment, weaker progress
2. weaker alignment, stronger progress
3. add explicit state deviation penalty

Goal:

- check whether TAL is over-imitation and becoming too conservative

### Round 3: generalization

If any TAL variant achieves stable laps:

- test on alternate track
- compare degradation versus `cth_v1`

## Important Constraints

Do not change these shared interfaces unless necessary:

- `compute_reward_and_termination(...)`
- `build_teacher_policy()`
- teacher action dict schema
- `info` fields for applied and teacher actions

Reason:

- the CTH branch already depends on the shared reward plumbing

## Parallel Run Isolation

Current code has been adjusted so TAL can reduce interference with the parallel `cth` branch, but there are still practical rules to follow.

Code-level isolation now in place:

- TAL and CTH can use different `TORCS_PORT` values
- `TorcsEnv` no longer uses global `pkill torcs`; it only stops the TORCS process started by the current env instance
- default `latest summary` output is disabled unless `TORCS_WRITE_LATEST_SUMMARY=1`
- TAL defaults to no automatic bootstrap from `baseline_tuned`
- TAL checkpoints remain under `checkpoints/<run_name>/`
- TAL summaries remain under `runs/<run_name>/`

Recommended TAL launch convention:

```bash
TORCS_PROFILE=tal_v1 \
TORCS_RUN_NAME=tal_v1_<tag> \
TORCS_CHECKPOINT_RUN_NAME=tal_v1_<tag> \
TORCS_PORT=3101 \
TORCS_BOOTSTRAP_BASELINE_TUNED=0 \
python test.py
```

Recommended parallel CTH launch convention:

```bash
TORCS_PROFILE=cth_v1 \
TORCS_RUN_NAME=cth_v1_<tag> \
TORCS_CHECKPOINT_RUN_NAME=cth_v1_<tag> \
TORCS_PORT=3201 \
python test.py
```

Hard limitation:

- same-machine true parallel execution is still only safe if the TORCS runtime itself accepts separate server ports cleanly in your local setup
- if port-level isolation does not work in practice, run TAL and CTH sequentially on the same host, or on separate machines/sessions
- interface churn will make branch integration harder

## Suggested Env Vars

Baseline TAL run:

```bash
TORCS_PROFILE=tal_v1
TORCS_TAL_TEACHER=heuristic
TORCS_ENABLE_SAFETY_SHELL=1
```

Useful tuning knobs:

```bash
TORCS_TAL_REWARD_SCALE
TORCS_TAL_STEER_WEIGHT
TORCS_TAL_ACCEL_WEIGHT
TORCS_TAL_BRAKE_WEIGHT
TORCS_TAL_PROGRESS_SCALE
TORCS_TAL_STEP_PENALTY
TORCS_TAL_DAMAGE_SCALE
TORCS_TAL_LAP_BONUS
TORCS_TAL_TEACHER_BASE_SPEED
TORCS_TAL_TEACHER_ANGLE_SPEED_SCALE
TORCS_TAL_TEACHER_TRACK_SPEED_SCALE
```

## Decision Rule

Keep TAL as a main candidate only if it beats CTH on at least one of:

- earlier stable lap completion
- fewer high-speed failures at similar average speed
- better transfer to another track

Otherwise TAL should be treated as:

- pretraining reward
- auxiliary reward
- teacher-guided stabilization phase before switching to CTH-style final optimization
