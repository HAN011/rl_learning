# Two-Wheel Balancer Curriculum Learning Experience

## 1. 这次追问聚焦的主题

这轮问题主要集中在 `curriculum_stage` 的理解和使用方式上，而不是奖励函数或机器人动力学本身。核心关注点有五个：

- `curriculum_stage` 如果不指定，默认到底是几。
- 训练时是不是应该从 `Stage 0` 开始，按顺序逐步推进。
- 当前 stage 设计是“逐级变难”，还是“每个 stage 只专注一个随机化因素”。
- 代码里究竟是哪一段在根据 `curriculum_stage` 改变训练环境。
- `--curriculum_stage` 这个命令行参数到底在做什么，依赖哪个库来解析。

---

## 2. 我确认下来的关键结论

### 2.1 默认 stage 是 0

如果不显式指定 `--curriculum_stage`，配置会回到默认的 `Stage 0`。

原因有两层：

- `TwoWheelBalancerCfg.env` 里的 `_curriculum_stage` 默认值是 `0`。
- 命令行参数 `--curriculum_stage` 没有设置默认值，因此不传时不会覆盖配置。

这意味着：

- `train.py` 不传 `--curriculum_stage` 时，会使用 easy-mode。
- `play.py` 不传时，会先尝试从 metadata 或路径推断；如果推断不到，最后还是回到 `Stage 0`。

---

### 2.2 课程训练顺序应该是 0 -> 1 -> 2 -> 3 -> 4

当前设计更适合按下面顺序推进：

`Stage 0 -> Stage 1 -> Stage 2 -> Stage 3 -> Stage 4`

这样做的原因是：

- `Stage 0` 先让策略学会最基本的平衡。
- 后续每一级只增加少量难度，避免一开始就把随机化全开。
- 如果后一级从前一级 checkpoint 继续训练，策略更容易稳定收敛。

这也说明课程训练不是“从任意 stage 单独起跑”，而是“逐级接力”更合理。

---

### 2.3 当前 stage 设计是“逐级累加变难”，不是“单因素隔离实验”

当前 `curriculum_stage` 的含义是累加式的：

- `Stage 0`：全部关闭
- `Stage 1`：只开启 friction randomization
- `Stage 2`：`Stage 1 + base mass randomization`
- `Stage 3`：`Stage 2 + com displacement randomization`
- `Stage 4`：`Stage 3 + observation noise`

所以这是典型的 curriculum 设计：

- 后一阶段会保留前一阶段已经获得的鲁棒性目标。
- 它不是 ablation，不是“每个 stage 只测试一个变量”。

如果想研究单因素影响，那应该单独做一套“只开 friction / 只开 mass / 只开 com / 只开 noise”的消融实验，而不是复用现在这套 stage 逻辑。

---

## 3. 代码调用链的理解

### 3.1 真正改变环境的是配置类，不是训练循环本身

环境变化的核心不在 `train.py` 的主循环里，而在配置类内部：

1. `cfg.env.curriculum_stage = N`
2. 触发 `curriculum_stage` 的 setter
3. setter 再调用 `_apply_curriculum_stage(N)`
4. `_apply_curriculum_stage()` 按 stage 修改：
   - `domain_rand.randomize_friction`
   - `domain_rand.randomize_base_mass`
   - `domain_rand.randomize_com_displacement`
   - `noise.add_noise`
   - `noise.noise_level`

也就是说，训练环境的变化首先是“配置对象被改写”，然后环境实例化时读取这份配置。

### 3.2 从命令行到环境生效的路径

命令行调用：

```bash
python humanoid/scripts/train.py --curriculum_stage 3
```

大致会经过这条路径：

1. `helpers.get_args()` 注册并解析 `--curriculum_stage`
2. `task_registry.make_env(...)`
3. `helpers.update_cfg_from_args(...)`
4. `env_cfg.env.curriculum_stage = args.curriculum_stage`
5. `TwoWheelBalancerCfg._apply_curriculum_stage(...)`
6. 环境按更新后的配置创建

这让我更清楚地区分了两件事：

- `train.py` 负责组织流程。
- `TwoWheelBalancerCfg` 负责决定环境长什么样。

---

## 4. 对命令行参数机制的新理解

下面这段代码：

```python
{
    "name": "--curriculum_stage",
    "type": int,
    "help": "Curriculum stage for env randomization. Overrides config file if provided.",
}
```

它的作用不是“从字符串名字里提数字”，而是：

- 声明一个命令行参数，参数名叫 `--curriculum_stage`
- 指定这个参数后面的值要按 `int` 解析
- 让解析后的结果出现在 `args.curriculum_stage` 里

依赖的库是 Isaac Gym 的：

- `isaacgym.gymutil`

由 `gymutil.parse_arguments(...)` 统一解析这些参数定义。它的使用风格和 Python 标准库 `argparse` 很像，但这里实际调用的是 Isaac Gym 自带的封装。

---

## 5. 这次整理后形成的稳定认知

经过这轮追问，我对这个项目里的 curriculum 机制形成了比较稳定的理解：

- `curriculum_stage` 是环境难度开关，不是奖励开关。
- 默认值是 `0`，也就是 easy-mode。
- 当前设计是累加式课程训练，不是单因素实验。
- 最合理的训练顺序是从 `Stage 0` 开始逐级推进。
- 命令行参数只是入口，真正起作用的是配置类里的 setter 和 `_apply_curriculum_stage()`。
- `play.py` 的 stage 对齐问题很重要，因为测试条件如果和训练条件不一致，会直接造成误判。

---

## 6. 对自己后续工作的提醒

- 以后看到“stage / curriculum / class”这类词，先确认它是“累加难度”还是“互斥分组”。
- 以后分析训练环境变化时，优先看配置类和参数覆盖链，不要只盯 `train.py` 主函数。
- 以后新增 CLI 参数时，要同时想清楚三件事：
  - 参数在哪里注册
  - 参数在哪里写回配置
  - 参数是否需要在 `play.py` 中和训练条件保持一致
- 以后做鲁棒性训练时，优先保持“训练条件”和“验证条件”严格对齐，否则会重复踩坑。
