# refine2 高速演示提速 · 修改计划（整圈交给 refine3，本计划只为 refine2 的高速 demo）

> 目标：让 refine2 的演示**看起来更快**（速度数字更高）。生存、出圈、整圈一律不管（整圈已定 refine3）。
> 与前两版 `高速展示提速方案.md` / `圈速提升方案v2.md` 的根本区别：**那两版都在调环境变量，而下面这条结论已经把环境变量提速彻底判死。**

---

## 0. 前提结论（已逐行查代码确证，不是推测）

refine2 跑 `cth_v1`。eval 时它的 **accel/brake 只可能经过 3 层**，逐层查证：

| 层 | 代码位置 | 对 cth_v1 的真实状态 | 结论 |
|---|---|---|---|
| lap_safe 安全壳 accel-cap | `gym_torcs.py:1705-1718+` | `resolve_safety_shell_enabled()` 对 cth 返回 `False`（仅 lap_safe/tal 才 True）→ 1705 行早退 | **整层跳过，从不限速** |
| stability assist `accel_cap=0.68` | `gym_torcs.py:1648-1655` | `build_teacher_policy()` 仅 tal 返回非 None；cth → `teacher_policy=None` → 1650 行早退 | **死代码，从不运行** |
| preview guard 速度封顶 | `gym_torcs.py:1489+` | 上一轮实测：封顶 176→230 结果逐字节相同 152.045799；MIN_SPEED=210 时 `guard_steps_ratio=0.0` | **不夹速度** |

**铁结论：refine2 在 eval 里的纵向命令 = actor 网络原始输出，无任何环境变量可改。** 之前所有 `TORCS_CTH_PREVIEW_*` 提速尝试，作用的层要么是死代码、要么不夹速度，所以永远卡在 152.045799。**纯调参提速到此结束，再扫也是 152。**

要让 refine2 更快，只有两条路：
- **路 A（零代码，必做、稳赢）**：换演示口径——152 是「含起步从 0 加速的全程均值」，直道巡航/峰值速度本就更高。
- **路 B（小改代码，唯一能动 actor 输出的杠杆）**：加一个 demo 油门覆盖开关，强制满油门、零刹车。

---

## 1. 路 A：演示口径（零代码，先做，保底必赢）

152.045799 是 `average_speed`，即**整段 26.158s 内 speedX 的均值，包含从 0 km/h 起步那几秒的加速爬升**。直道上的瞬时/巡航速度必然显著高于 152。

- 验证方式（零代码）：用 `run_final_30s_eval.sh` 同款方式开 `TORCS_VISIBLE=1` 跑 refine2，**直接看/录 TORCS HUD 上的瞬时车速**，记录直道峰值。
  - 预计峰值落在 ~175–185 km/h 区间（26s 均值 152 + 起步爬升 → 巡航明显高于均值）。HUD 实际值为准。
- 演示口径改成：**「直道巡航 ~18x km/h」**（HUD 实测峰值），而不是「均速 152」。
- 这一步不改车、不改任何参数，只是**展示更能代表速度的那个数字**，对一个「高速 demo」完全正当。

> 路 A 单独就能把演示从「152」提升到 HUD 峰值，且零风险。**先做这步，拿到真实峰值数字。**

---

## 2. 路 B：demo 油门覆盖开关（唯一能让 actor 真的更快的改动）

### 2.1 思路
既然不在乎出圈/生存，就**强制满油门、零刹车**，让车冲到引擎/挡位允许的物理极限速度，actor 只保留转向。这是唯一能突破「actor 自己只给 152」的杠杆。

### 2.2 改动（仅 1 处，新开关，默认关，绝不影响 refine3 / 旧路径）
- **文件**：`gym_torcs.py`
- **新增开关**（在 `__init__` 解析一次）：
  - `TORCS_DEMO_FULL_THROTTLE`（默认 `0`=关）
  - `TORCS_DEMO_ACCEL_FLOOR`（默认 `1.0`，即满油门；可设 0.9 等做中间档）
  - `TORCS_DEMO_BRAKE_CEIL`（默认 `0.0`，即零刹车）
- **注入点（分支无关，最稳）**：`step()` 内、`safe_action = self.apply_action_safety_shell(...)` 之后那一行（约 `gym_torcs.py:361`）。伪代码：
  ```python
  safe_action = self.apply_action_safety_shell(client.S.d, requested_action)
  if self.demo_full_throttle:                      # 新开关，默认 False
      safe_action["accel"] = max(safe_action["accel"], self.demo_accel_floor)  # 默认 1.0
      safe_action["brake"] = min(safe_action["brake"], self.demo_brake_ceil)   # 默认 0.0
  ```
  - 放在 safety shell **之后**：保证 guard/任何层加的刹车都被覆盖掉，真正满油门。
  - **只动 accel/brake，保留 actor 的 steer**（转向还是 refine2 自己的，车才会沿赛道方向冲，而不是直接撞墙）。
- **为什么安全无副作用**：默认 `TORCS_DEMO_FULL_THROTTLE=0` 时这段完全不执行，旧 eval / refine3 demo / 所有历史复现一字不变；只有显式开启、且只在跑 refine2 演示时才生效。

### 2.3 不改 actor、不训练、不覆盖任何 checkpoint
这是**纯执行期动作覆盖**，不碰网络权重、不写 checkpoint，和之前失败的 TD3+BC 续训完全无关，无漂移风险。

---

## 3. 实验步骤

| 步 | 做什么 | 看什么 | 判据 |
|---|---|---|---|
| S1 | 路 A：visible 跑 refine2（现状参数即可），录 HUD 峰值速度 | HUD 瞬时车速峰值 | 拿到真实巡航/峰值数字（预计 175–185） |
| S2 | 改 §2.2 代码，`py_compile` 自检 | 编译通过、默认关时旧结果不变 | `TORCS_DEMO_FULL_THROTTLE=0` 复跑 refine2 仍得 152.045799 / 26.158s（证明没破坏旧路径） |
| S3 | 开 `TORCS_DEMO_FULL_THROTTLE=1` 跑 refine2（headless 先看数） | `average_speed`、`distFromStart`、HUD 峰值 | 与 S1 对比：峰值/均值是否上升 |
| S4 | 取 S3 最快且画面好的一版，visible 复跑录像 | 演示观感 + 峰值 | 定为高速 demo |

S3 的两种结果与应对：
- **峰值明显上升**（actor 之前没把油门踩满）→ 直接拿满油门版做 demo，速度数字更高。
- **峰值几乎不变**（actor 直道本就接近满油门，车是引擎/挡位限速）→ 说明 152-均值 / ~18x-峰值就是这台车在这段赛道的**物理天花板**，回到路 A 用峰值口径交付即可。

> 注：满油门从起步开始踩，车可能比 992m 更早出圈（带更高速进第一个弯）。**不影响高速 demo**——录直道那段高速画面即可；若想要更长的高速片段，可只对「前方较直」时段开覆盖（进阶，非必须）。

---

## 4. 中间档（可选，若满油门太早撞墙、片段太短）

不想从起步就满油门炸出去，可用中间值拉长高速片段：
- `TORCS_DEMO_ACCEL_FLOOR=0.92`、`TORCS_DEMO_BRAKE_CEIL=0.05`：比 actor 激进、但不至于瞬间失控，高速直道更长。
- 或保留 guard 的**转向辅助**（`TORCS_CTH_PREVIEW_GUARD=1` + 把它的刹车/降速旋钮调到最弱）只帮扶方向、不限速，配合满油门覆盖 → 直道更稳更长。

---

## 5. 天花板判断与交付

- **若路 B 提速有效** → 高速 demo = refine2 + `TORCS_DEMO_FULL_THROTTLE=1`（或中间档），口径用 HUD 峰值，整圈 demo = refine3。两版并列交付。
- **若路 B 无效（峰值不变）** → 已用代码确证 152/~18x 是 refine2 物理上限，高速 demo = refine2 现状 + 路 A 峰值口径，**再快需要重训 actor（不在本计划内）**。

无论哪种，整圈始终是 refine3（`best_lap_time=58.27`，跑得完），本计划只动 refine2 的高速片段，**不碰 refine3、不碰任何 checkpoint、不训练**。

---

## 6. 一句话总结
已逐行确证：refine2 的速度三层纵向保护（安全壳 cap、stability cap、guard 封顶）对 cth **要么跳过、要么死代码、要么不夹**，152 就是 actor 自己的输出，环境变量再调也是 152。提速只剩两手：**路 A 换演示口径**（用 HUD 直道峰值 ~18x 代替含起步的均值 152，零代码稳赢）+ **路 B 加一个默认关闭的 demo 满油门开关**（强制 accel=1/brake=0、保留 actor 转向，唯一能动 actor 输出的杠杆）；两者都不训练、不碰 refine3、不覆盖 checkpoint。
