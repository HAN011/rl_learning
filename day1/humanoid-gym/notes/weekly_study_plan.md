# 未来一周学习计划：从二轮平衡车到人形机器人 RL

## Context

你是一个 RL 初学者，已经完成了以下工作：
- 在 humanoid-gym 上调通了 two_wheel_balancer 的基本训练流程
- 理解了 URDF 物理定义、奖励设计、域随机化、课程训练的基本概念
- 发现了 `init_at_random_ep_len` 对训练指标的污染问题，已修复
- 实现了 curriculum_stage 0-4 的配置切换机制
- 建立了独立评估脚本 eval.py，用真实 reset 的 deterministic rollout 评估
- **修复了 empirical_normalization 问题（现在二轮平衡车默认关闭）**
- **大量超参数搜索（noise、residual RL、entropy、ns）全部 timeout_ratio=0.0**
- **关键诊断：参考控制器几乎不改变倾倒轨迹，说明控制链路本身有问题**
- 每天可投入 4-6 小时

目标：一周后具备独立设计 RL 训练环境、调参、评估的能力，并对人形机器人 RL 有初步了解。

**当前 blocker**：控制动作无法产生预期的物理稳定效果，原因未定位。
**不应该改奖励函数**：奖励设计不是问题，控制链路才是。

---

## Day 2（周二）：诊断控制链路——找到真正的物理 bug

**核心目标**：确认"给两个轮子施加正力矩，小车会向前走"这个最基本的假设是否成立。

**上午（2h）- 诊断脚本**

让 Codex 写一个最小诊断脚本 `humanoid/scripts/diagnose_control.py`，用 Isaac Gym 跑单环境，
**绕开 PPO 和奖励**，直接控制动作并观察物理响应：

```python
# 脚本逻辑（伪代码）
for step in range(20):
    # 阶段 1：零动作，记录自然倾倒速度
    action = [0.0, 0.0]

for step in range(20):
    # 阶段 2：正力矩，观察 pitch/position/wheel_vel 变化
    action = [+1.0, +1.0]  # 最大正力矩

for step in range(20):
    # 阶段 3：负力矩，观察反向效果
    action = [-1.0, -1.0]

# 每步打印：step, pitch, pitch_rate, x_pos, wheel_L_vel, wheel_R_vel, torque_L, torque_R
```

**关键验证点**：
1. 正力矩时，左右轮角速度符号是否相同（同向旋转）？
2. 正力矩时，x 坐标向哪个方向移动？
3. 当小车向前倾（pitch > 0）时，正力矩是否能让 pitch 减小？

**下午（3h）- 根据诊断结果修复**

根据诊断输出，修复对应问题：

| 诊断结果 | 修复方向 |
|---|---|
| 左右轮角速度符号相反 | 再次检查 URDF 轴向，或在代码里对左轮动作取反 |
| 正力矩让 x 向后走 | 参考控制器 u 符号取反 |
| pitch > 0 时正力矩不减小 pitch | 检查 `_compute_ref_actions` 中 pitch 的定义轴 |
| 任何动作都几乎不改变物理状态 | 检查 `_compute_torques` 是否正确应用到仿真器 |

**产出**：诊断报告 + 至少一个明确的物理 bug 被修复 + 参考控制器开环测试 mean_len > 2s

---

## Day 2（周二）：巩固 Stage 0 + 奖励工程实验

**上午（2h）- 理论**
- 学习奖励工程（Reward Shaping）的基本原则：
  - 稀疏奖励 vs 稠密奖励
  - 奖励 scale 的影响
  - 常见陷阱：reward hacking、奖励间冲突
- 回顾项目中 `_reward_upright`、`_reward_stability`、`_reward_energy` 等函数的设计意图

**下午（3h）- 实操**
- 如果 Day 1 的 Stage 0 已收敛（timeout_ratio > 50%），做 3 组奖励对比实验（每组 200 迭代）：
  1. 调大 `upright` 权重到 10.0，观察行为变化
  2. 去掉 `energy` 惩罚，看策略是否变得"暴力"
  3. 调大 `termination` 惩罚到 -200，看是否加速学习
- 如果 Day 1 的 Stage 0 未收敛，优先继续调参（学习率、探索噪声、gamma 等）
- 所有评估用 eval.py

**产出**：3 组实验对比表 + 对奖励权重影响的定性总结（或 Stage 0 收敛方案）

---

## Day 3（周三）：课程训练实战 Stage 0 → Stage 2

**上午（1h）- 理论**
- 学习课程训练（Curriculum Learning）的核心思想：
  - 为什么先易后难有效（loss landscape 角度）
  - 什么时候该升级难度（基于性能阈值 vs 基于迭代数）
  - 过早升级 vs 过晚升级的后果

**下午（4h）- 实操**
- 确认 Stage 0 模型已经达标（timeout_ratio > 50%）
- 从 Stage 0 最优 checkpoint 继续训练 Stage 1（开启摩擦随机化）
  ```bash
  python train.py --curriculum_stage 1 --resume --load_run <stage0_run> --run_name stage1 --max_iterations 300
  ```
- 用 eval.py 验证 Stage 1 模型
- 如果 Stage 1 通过，继续 Stage 2（加入质量随机化）
- 记录每个 stage 的升级时机和 timeout_ratio 变化

**产出**：Stage 0 → Stage 1 → Stage 2 的训练日志和 eval 结果

---

## Day 4（周四）：观测空间与状态表示

**上午（2h）- 理论**
- 学习 RL 中观测空间的设计：
  - 马尔可夫性（Markov Property）：为什么需要 frame_stack
  - privileged observation vs policy observation（teacher-student）
  - 观测归一化的重要性
- 对照代码 `compute_observations()` 和 `obs_names` 配置

**下午（3h）- 实操**
- 完成 Stage 2 → Stage 3 → Stage 4 的课程训练
- Stage 4 加入观测噪声后，观察策略性能下降多少
- 实验：把 `frame_stack` 从 3 改成 1，看策略能否学会（理解历史信息的作用）
- 最终用 Stage 4 模型跑 play.py 可视化，观察策略行为

**产出**：完整 Stage 0-4 课程训练完成 + frame_stack 对比实验

---

## Day 5（周五）：Sim2Real 基础 + Isaac Gym 深入

**上午（2h）- 理论**
- 学习 Sim-to-Real 的核心概念：
  - Reality Gap：仿真和真实之间的差异来源
  - Domain Randomization：为什么随机化能帮助跨域迁移
  - System Identification：如何让仿真更接近真实
- 推荐论文：OpenAI "Learning Dexterous In-Hand Manipulation"（不用精读，看 method 部分）

**下午（3h）- 实操**
- 深入阅读 Isaac Gym 的环境代码：
  - `base_task.py`：理解仿真循环 step → compute_obs → compute_reward → check_termination → reset
  - `legged_robot.py`：理解 URDF 加载、contact force 获取、domain randomization 实现
- 给当前项目加一个新的域随机化：`push_robots = True`，实现外部推扰
  ```
  Stage 5: Stage 4 + push_robots, max_push_vel_xy = 0.3
  ```
- 验证策略在推扰下的表现

**产出**：对 Isaac Gym 仿真循环的完整理解 + Stage 5 推扰实验

---

## Day 6（周六）：人形机器人 RL 调研

**上午（3h）- 论文 + 项目调研**
- 阅读 2-3 篇人形机器人 RL 的代表性工作：
  1. **Humanoid-Gym 原始论文**（如果有的话）或 RobotEra 的相关论文
  2. **Berkeley Humanoid**（https://berkeley-humanoid.com）- UC Berkeley 的人形机器人 RL
  3. **LeRobot by HuggingFace** 或 **Isaac Lab** - 了解当前主流框架
- 关注这些项目和你当前项目的区别：
  - 观测空间有多大（几十维 vs 几百维）
  - 动作空间（位置控制 vs 力矩控制）
  - 奖励函数设计（步态模仿 vs 纯任务驱动）
  - 训练规模（几千环境 vs 几万环境）

**下午（2h）- 对比整理**
- 把二轮平衡车和人形机器人的 RL pipeline 做对比表
- 识别共同点（都用 PPO、都需要课程训练、都需要 sim2real）
- 识别差异（维度、复杂度、步态生成、接触建模）

**产出**：人形机器人 RL 调研笔记 + 与本项目的对比表

---

## Day 7（周日）：回顾与规划

**上午（2h）- 知识整理**
- 把一周的实验和笔记整理成一份完整的学习记录
- 按"问题 → 知识点 → 适用条件"格式归档（不用按时间线）
- 更新 notes/ 目录

**下午（3h）- 下一步规划**
- 尝试恢复项目中已删除的人形机器人环境（XBot URDF），或从 Isaac Lab 获取一个简单的人形模型
- 搭建人形机器人的最小可运行环境（哪怕只是站立任务）
- 制定下一周的目标：人形机器人的站立 → 行走

**产出**：一周总结文档 + 人形机器人环境初步搭建

---

## 关键原则

1. **每天先理论再实操**，理论不超过 2 小时，避免"只看不做"
2. **每次实验只改一个变量**，否则无法归因
3. **所有评估用 eval.py**，不看训练日志里的 timeout_ratio
4. **实验结果当天记录**，不要攒到最后写
5. **卡住超过 30 分钟就换方向**，不要死磕一个超参数

## 验证方式

- Day 3 结束时：Stage 0-2 课程训练跑通，有真实 eval 数据
- Day 5 结束时：Stage 0-4 + 推扰全部跑通
- Day 7 结束时：有人形机器人 RL 的初步调研和下一步规划
