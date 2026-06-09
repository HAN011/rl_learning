# TORCS 赛车性能迭代计划

## 目标
- 在现有 `DDPG_Torcs_PyTorch` 项目上进一步提升赛车表现。
- 优先减少 `out_of_track`，其次提升圈速、平均速度和单圈完成能力。
- 不再基于当前 `full_opt` 继续盲调，而是以 `baseline_tuned` 为稳定起点做新一轮迭代。

## 当前结论
- `baseline_tuned` 是当前最稳的起点，适合继续续训和做结构性优化。
- `baseline_tuned_push1` 说明“更激进的策略可以更快”，但评估时容易 `out_of_track`，说明速度和稳定性没有被同时约束。
- `full_opt` 当前训练结果明显异常，不适合作为继续优化基础。
- `gym_torcs.py` 的 improved termination 逻辑里，`np.any(track < 0)` 很可能过于激进，可能会把可恢复状态误判成出界。
- `test.py` 当前只记录 `reward / average_speed / total_steps / termination_reason / completed`，没有直接记录圈速、单圈完成情况、总行驶距离等核心指标。

## 任务边界
- 优先修改 `DDPG_Torcs_PyTorch/gym_torcs.py` 和 `DDPG_Torcs_PyTorch/test.py`。
- 如有必要，再修改 `DDPG_Torcs_PyTorch/ActorNetwork.py`、`DDPG_Torcs_PyTorch/CriticNetwork.py`。
- 不要覆盖现有 `baseline`、`baseline_tuned`、`full_opt` profile。
- 新增 profile，例如 `lap_safe_v1`、`lap_safe_v2`。
- 第一阶段先在现有 DDPG 框架内完成优化，不立即切换新算法。

## 第一阶段：补齐评估与模型选择
- 在 episode 日志中新增字段：`distRaced`、`distFromStart`、`curLapTime`、`max_abs_trackPos`、`mean_abs_trackPos`、`mean_abs_angle`、`damage_sum`。
- 如果 TORCS 可直接提供单圈时间或圈数，则直接记录；如果没有，则基于 `distFromStart` 的回绕行为推断圈完成。
- 在训练过程中增加固定频率评估，例如每 `10` 个 episode 做一次短评估。
- 增加 best checkpoint 机制，不再只保存“最后一轮模型”。
- best checkpoint 的选择顺序设为：`out_of_track` 更少 > 完成圈数更多或 `distRaced` 更高 > `lap_time` 更短 > `average_speed` 更高。

## 第二阶段：重做 reward 和 termination
- 以 `baseline_tuned` 为基础新建 `lap_safe_v1`。
- reward 主体改为“前进距离增量”或更接近 `distRaced` 增长的形式，不再只依赖 `speed * cos(angle)`。
- 保留软惩罚：`abs(trackPos)`、`abs(sin(angle))`、`damage_delta`、`abs(delta_steer)`。
- 增加边缘区惩罚，当 `abs(trackPos) > 0.7` 时加重惩罚，但不要立刻终止。
- 将硬出界条件改为更保守的版本，例如 `abs(trackPos) > 1.05` 或连续多帧明显离路后才终止。
- 去掉或弱化 `np.any(track < 0)` 这类容易误判的单帧硬终止条件。
- 增加单圈完成 bonus，使策略真正优化“跑完且更快”，而不是只追求短时高 reward。

## 第三阶段：动作安全壳
- 对 `accel` 和 `brake` 做互斥或冲突抑制，避免同时大油门和大刹车。
- 引入基于速度的转向限制，速度越高，允许的最大 `steer` 越小。
- 当 `abs(trackPos)` 较大或 `angle` 偏差较大时，动态降低 `accel` 上限。
- 在高风险状态下加入轻量 brake assist，但不要完全替代策略动作。
- 对 `steer` 做一阶平滑，减少抖动和连续急修方向。

## 第四阶段：训练流程优化
- 从 `baseline_tuned` checkpoint 续训，而不是从零开始。
- 保留探索，但让 `steer` 噪声、`accel` 噪声、`brake` 噪声分开衰减。
- 优先调小 `steer` 噪声，减少高速时的横向失控。
- 只对高价值参数做小范围 sweep：`steer_sigma`、`gamma`、`tau`、`stochastic_brake_prob`。
- 每个候选都执行相同评估协议，不能只看训练 reward。

## 第五阶段：必要时做结构升级
- 如果前四阶段仍然无法同时提升“圈速”和“稳定性”，再进入结构升级。
- 候选升级 1：补充状态特征，例如上一时刻动作、`delta_steer`、局部 track sensor 统计量、左右赛道宽度差、简易曲率特征。
- 候选升级 2：修正 critic 为标准标量 Q 输出，并检查 actor-critic 更新逻辑是否标准。
- 候选升级 3：如果允许中等规模重构，优先考虑迁移到 TD3，而不是继续深拧当前 DDPG。

## 实验协议
- Smoke test：每个新 profile 先跑 `5` 个 episode，确认不会明显崩溃。
- Short run：`30` 个 episode，至少 `3` 个 seed，先看出界率和趋势。
- Full run：对通过 short run 的方案跑 `150` 到 `200` 个 episode。
- 评估顺序：先 `g-track-1`，再 `g-track-2` 零样本测试。
- 所有比较都以当前 `runs/baseline_tuned/eval/episode_summary.csv` 作为稳定性基线。

## 验收标准
- `g-track-1` 上的 `out_of_track` 次数不能高于 `baseline_tuned`。
- 在稳定性不下降的前提下，`distRaced`、单圈完成能力或圈速至少有一项明显提升。
- `g-track-2` 零样本测试不能出现明显退化，否则该方案不能作为最终版本。
- 最终交付的不应是“训练 reward 最高”的模型，而应是“稳定性优先、速度次优”的 best checkpoint。

## 预期交付物
- 修改后的代码。
- 新增 profile 与对应 checkpoint。
- 新的 `runs/<profile>/train|eval/episode_summary.csv`。
- 一份结果对比总结，至少包含稳定性、圈速或替代指标、平均速度、终止原因分布。
- 最终推荐一个可交付 profile，并说明为什么它优于 `baseline_tuned`。

## 建议实施顺序
- 先做“指标补齐 + best checkpoint + reward/termination 修正”。
- 再做“动作安全壳 + 小范围调参续训”。
- 只有前两步见顶后，再考虑 critic 修正或 TD3 迁移。
