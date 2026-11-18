# CPU 配方：上游 CPU workflow 在 h2r_bench 任务上的规划

CPU 配方把上游 `dev/cpu`（`3d132978`）的 CPU workflow 接到 bench 的共同任务上：sampling MPC 在 MuJoCo 的 CPU 批量接口（`mujoco.batch`）上运行，每个控制帧重规划，物体滑脱时退回重新规划。它是本 fork 的正式配方；原有的 fast MJWarp 配方（[h2r.md](h2r.md)）保留为可选配方，两者的产物不能混用。

```text
task ──► FlashCHORD prepare（V2D Sharpa IK）──► robot_reference.npz
task + 该参考 ──► spider.h2r.prepare --reference-from ──► prepared
task + prepared ──► spider.h2r.cpu_plan ──► plan（每个控制帧的关节位置目标）
task + plan ──► spider.h2r.evaluation ⇄ bench 评估 ──► rollout ──► spider.h2r.check
```

规划的循环、恢复逻辑和 pass 判定是上游的 `examples/run_cpu.py` 与 `spider/postprocess/evaluate_cpu.py`，文件与上游逐字节相同。本 fork 提供的是 bench 任务上的私有世界和封存。部署方式仍是 `offline-control-replay`：每条序列规划一次，全部评估 seeds 回放同一份控制。上游在自己的场景上的通过率不能外推到这里，原因见[与上游的差异](#与上游的差异)。

## 环境

规划只用 CPU，需要 MuJoCo ≥ 3.17。它有独立的环境，不改动 fast MJWarp 配方的环境（MuJoCo 3.7）：

```bash
bash scripts/h2r_cpu_environment.sh /path/to/spider-cpu-env   # 容器内为 /opt/h2r/spider-cpu
```

环境固定 MuJoCo 3.17.0 与 Torch 2.7.1，完整约束见 `scripts/h2r_cpu_runtime_constraints.txt`。上游文档的依赖清单漏了 `matplotlib`（Viser 面板用到），脚本已补上。参考导入、规划、回放客户端和回放核验都在这个环境中运行；参考本身由 FlashCHORD 的 prepare 在它自己的环境中生成。

## 参考

上游的 CPU workflow 没有 IK 步骤，它的场景带着 FlashCHORD 所用的 V2D Sharpa 重定向。这里同样使用这套参考，而不是 SPIDER 的 Mink IK。先按 h2r_bench 的 FlashCHORD 方法页对同一个 task 运行 `flash_chord.h2r.prepare`（已有时直接复用），再导入：

```bash
python -m spider.h2r.prepare --task "$TASK" --reference-from "$FC_PREPARED" --output "$PREPARED"
```

导入不运行任何 IK。以下任一项不满足即拒绝：

- FlashCHORD 的 binding 格式受支持，`task_id`、`demo_id`、`robot_id` 与 `--task` 相同。
- `robot_reference.npz` 的 hash 与该 binding 记录的一致。
- 关节名集合与 task 的机器人相同，时间网格与 demo 相同，数值有限且在关节限位内。
- 腕部旋转角在相邻帧之间的变化小于 π（V2D 的腕部分解超出限位时回绕 2π，关节坐标不能跨过这种跳变插值）。

| prepared 文件 | 内容 |
| --- | --- |
| `trajectory_kinematic.npz` | demo 时间网格上的完整 `qpos`（机器人取导入的关节参考，物体取 task 参考）与 `ctrl` |
| `initial_robot.json`、`initialization.json` | 第 0 帧的具名机器人初态与方法侧预检结果（与 Mink 路径相同的预检） |
| `reference_source.json` | 参考种类、FlashCHORD binding 与 `robot_reference.npz` 的 hash、该数据集的 V2D IK 设置 |
| `h2r_binding.json` | 身份、`reference: v2d-sharpa-ik`、方法版本、文件 hash |

导入的参考不经过 SPIDER 的 3 帧平均等参考处理。预检不通过时与 Mink 路径一样：prepare 照常完成，规划只封存初态，评估记为 `initial_collision`。

支持的数据集取决于 FlashCHORD 的 IK 配方：ARCTIC、OakInk2、HOT3D、TACO、DexterHand。

## 规划

```bash
python -m spider.h2r.cpu_plan --prepared "$PREPARED" --task "$TASK" \
  --seed 0 --threads 16 --output "$PLAN"
```

同一 task、prepared、seed 和设置给出同一份 plan，与线程数无关。规划分四步：

1. **私有模型**（`cpu_scene.private_model`）：由 `task.scene_xml` 在内存中编译，只改物理步长。以 `dampratio` 给出的执行器阻尼在编译时换算，结果随 MuJoCo 版本变化（3.17.0 相对 3.11 最多差约 1 %）；task 的 `compiled.json` 记录了评估所用的值，这里按该记录设置 `kv`，并核对刚度、关节摩擦、阻尼和 armature 与记录一致。
2. **表**（`cpu_scene.tables`）：上游 `scene.npz` 的那组表，按关节名从 task 填出。第 `i` 行是参考在第 `i` 个控制帧**结束**时刻的值；初态是参考在 0 s 的值，单独给出。
3. **规划**：调用上游 `run_cpu.track`，simulator 模块为 `cpu_world`。
4. **封存**：每个控制帧实际下发的关节位置目标（滤波、限幅之后）写入 `controls.npz`。封存前在单个私有 world 中从初态回放这些目标，必须复现规划记录的状态（容差 1e-9；在真实任务上实测逐位相同）。

| 规划结果 | 产物 |
| --- | --- |
| 跑完全程 | `plan.json`，记录上游判定 `native.success`（1 或 0）。判定未通过的 plan 同样封存，成绩由 bench 评估决定 |
| 上游提前停止（物体丢失且回退用尽） | 没有 `plan.json`；`failure.json` 与截断的 `trajectory_cpu.npz` 作诊断，该任务计为失败 |
| 初态未通过预检 | `plan.json`，`deployment: initial-pose-only`，不含控制 |

plan 目录另有上游原生输出 `trajectory_cpu.npz` 与 `metrics.json`、实际配置 `resolved_config.yaml` 和表的副本 `scene_tables.npz`。`plan.json` 记录配方（`recipe: cpu`）、规划 backend（`mujoco-cpu`）、MuJoCo 版本、线程数、时钟、上游判定、回退次数和自检偏差。上游判定是在私有世界里对方法自己的参考计分，只作记录，不是成绩。

### 配置

解析顺序为上游 `examples/config/cpu.yaml`、H2R 适配层 `spider/h2r/config/h2r_cpu.yaml`、显式 `--config`，实际值写入 `resolved_config.yaml`。`--config` 是一个含 `mpc`（上游配方的算法字段）、`recovery`（上游的恢复设置）、`h2r`（适配层）三个可选映射的 YAML；任务、路径、设备、viewer 和运行长度不是可覆盖的设置，未知字段报错。降低预算只用于调试，不是配方。

| 设置 | 默认 | 来源 |
| --- | --- | --- |
| samples／每帧迭代 | 128／4 | 上游 `cpu.yaml` |
| horizon／knot／重规划 | 0.4 s／0.2 s／每个控制帧 | 上游 `cpu.yaml` |
| 噪声（腕部位置／腕部旋转／手指） | 0.01 m／0.01 rad／0.10 rad | 上游 `cpu.yaml` |
| 恢复 | 单帧加噪重规划至多 2 次；滑脱或丢失时退回 1 s、2 s、3 s 重新规划，重规划段 horizon 0.8 s | 上游 `cpu.yaml` |
| 控制帧／私有物理 | 0.05 s／0.01 s（5 子步） | 上游场景的时钟，写在 `h2r_cpu.yaml` |
| 残差滤波／限幅 | EMA 0.5；腕部位置 0.2 m，其余 1 rad | 上游场景中的数值（`chord.py` 的 `ACTION_EMA`、发布场景的 `action_clip`），写在 `h2r_cpu.yaml` |

## 评估与核验

评估客户端和核验与 fast MJWarp 配方是同一套入口，在 CPU 环境中运行：

```bash
python -m spider.h2r.evaluation --plan "$PLAN" --task "$TASK" --socket "$SOCKET" --output "$CONTROLLER_OUTPUT"
python -m spider.h2r.check --plan "$PLAN" --task "$TASK" --run "$ROLLOUT" --output "$REPLAY_CHECK"
```

客户端在 start 中声明 `control_dt_s = 0.05`；评估物理步长只从 `hello.dt_s` 得知，300 Hz 下每个目标保持 15 个物理步。controller 信息是 `plan.json` 的内容加上它的 sha256。在 bench 中声明评估时使用 `--variant cpu-offline-control-replay --training-backend mujoco`，与 fast MJWarp 配方的结果分属不同比较组；完整命令见 h2r_bench 的 `docs/methods/spider.md`。

## 方法内 rollout 的导出

bench 在独立评估之外，还报告方法在自己的仿真里的表现（`method-sim`，自报）。对 CPU 配方，这条 rollout 就是 plan 本身：

```bash
python -m spider.h2r.method_rollout --plan "$PLAN" --task "$TASK" --output "$METHOD_ROLLOUT"
```

入口在单个私有 world 中回放封存的目标：与规划相同的模型、步长和子步数，从评估初态（plan 提出的机器人姿态、物体取参考首帧、零速度）开始，没有辅助和扰动。回放须在 `1e-9` 内重现规划记录的状态（`trajectory_cpu.npz`），否则拒绝导出；MuJoCo 版本与规划时不同会在这里被发现。输出是 `h2r-method-rollout-v1` 包：每个物理步（100 Hz）的机器人关节、物体自由根位姿和物体关节，以及上游 `evaluate_cpu` 的判定（记在 plan 的 `native` 中）。不重新规划，约几秒完成。

初态未通过 SPIDER 预检、没有控制的 plan 导出一条在该姿态停留一个物理步后结束的 rollout，首帧交给 bench 的准入检查。包由 bench 的 `import-method-rollout` 导入并用共同指标计分，见 h2r_bench 的 `docs/methods/spider.md`。

## 与上游的差异

| 项目 | 上游 CPU workflow | 这里 | 性质 |
| --- | --- | --- | --- |
| 场景 | FlashCHORD 导出的 `scene.xml`：腕部 1600 N/m、手指 40 N·m/rad，铰链摩擦取最大重力矩的 1.5 倍，物体摩擦 1.0 | bench 的 task 场景：腕部 300 N/m、手指 0.9–13.2 N·m/rad，铰链参数逐序列设置，滑动摩擦 2.0 | 共同条件 |
| 腕部旋转 | 球关节，限幅比例力矩，逐子步重算；残差按 roll／pitch／yaw 复合到参考四元数 | 三个串联 hinge 的 position actuator；残差直接加在三个 hinge 目标上，不施加力矩 | 必要边界转换 |
| 腕部旋转代价 | qpos 中腕部四元数对参考的旋转向量 | 腕部 body 姿态对参考的旋转向量，权重相同 | 必要边界转换 |
| 表 | `chord_export.py` 从 FlashCHORD 环境导出 | `cpu_scene.tables` 按关节名从 task 与 prepared 构建 | 必要边界转换 |
| 参考 | FlashCHORD 环境在控制帧上的参考（位姿插值） | 同一套 IK 在 bench 任务上的结果，机器人按关节坐标线性插值到控制帧，物体旋转用符号对齐的 NLERP | 必要边界转换 |
| 行对应 | 第 `i` 行是第 `i` 帧开始时的姿态，执行结果比参考晚一帧 | 第 `i` 行是第 `i` 帧结束时的姿态 | 任务适配：评估按绝对时间计分 |
| 执行器阻尼 | 写在导出的场景里 | 按 task 的 `compiled.json` 设置 | 消除编译版本差异 |
| MuJoCo 版本 | 发布结果用 4.0.1 | 3.17.0 | 版本选择 |
| 提前停止的运行 | 保存到停止处的运动 | 不封存 plan | 沿用 H2R 约定 |

保留不变：残差结构、滤波与限幅数值、优化参数、奖励权重、恢复逻辑、判定阈值、控制与物理时钟。

对上游文件的改动只有一处：`spider/simulators/cpu.py` 的 `setup_env` 增加可选的 `model`、`scene` 参数（不传时行为不变），用于接收内存中的 task 模型与表。

## 检查与已验证范围

```bash
python -m unittest discover -s tests -p 'test_h2r_cpu*.py' -v   # CPU 环境
pytest spider/simulators/cpu_test.py spider/postprocess/evaluate_cpu_test.py spider/viewers/viser_viewer_test.py
```

`test_h2r_cpu.py` 用一个合成任务覆盖：参考按名称导入及各项拒绝，执行器阻尼取自 task 记录，表的名称映射与行对应，腕部旋转残差经滤波到达 hinge 目标，代价中的腕部项，状态恢复后逐位重跑，以及极小预算下的端到端封存、提前停止、初态未通过和配置拒绝。没有 `mujoco.batch` 的环境会跳过需要它的测试。其余 `test_h2r*.py` 在 fast MJWarp 的环境中运行。

### 上游对照（2026-11-18）

- 上游的 44 项测试（上面的 `pytest` 命令）在 CPU 环境（MuJoCo 3.17.0）中通过。
- 原生复现：上游源码（`3d132978`）在 MuJoCo 3.17.0、16 线程下运行上游发布的场景 `s01-box-use/1`（seed 0，默认配方），物体 0.73 cm、1.4°，铰链 0.5°，开合 5/5，判定通过，耗时 413 s。上游发布的结果由 4.0.1 生成，为 0.79 cm、1.3°、通过；两个版本的轨迹不逐位相同，判定一致。
- 本 fork 与上游源码在同一场景、同一版本下的输出逐位相同（42 个数组，不含计时）。

### 真实任务检查（2026-11-18）

suite `h2r-v1-6a7d7e01`，Sharpa Wave，记录在 bench 的 `runs/checks/20261118-spider-cpu/`。规划为默认配方、seed 0、16 线程，规划时机器负载很高，耗时偏长。评估为 bench 的 MuJoCo 3.11、300 Hz，`sim` 与 `sim-dr` 各两个 seed（0、1），默认扰动。这些检查说明链路可执行以及配方在共同条件下的初步表现，样本很少，不是正式成绩。

| 任务 | 规划：私有世界中的结果与上游判定 | bench SR（`sim`／`sim-dr`） | AUC10-All（`sim`／`sim-dr`） |
| --- | --- | --- | --- |
| ARCTIC `s01-box_use_01`（29.6 s，双手，铰接） | 599 s；物体 1.11 cm，开合 5/5，通过；无回退 | 0/2 ／ 0/2 | 18.0 %／17.9 % |
| OakInk2 `scene_02__A007…st01`（5.5 s，双手，两个刚体） | 164 s；物体 0.84 cm，通过；无回退 | 2/2 ／ 2/2 | 92.3 %／91.4 % |
| OakInk2 `scene_01__A005…st06`（7.3 s，右手） | 180 s；物体 0.82 cm，通过；无回退 | 0/2 ／ 0/2 | 89.5 %／87.7 % |
| HOT3D `P0001_a9d6c83d_seg017`（5.1 s，双手） | 256 s；物体 2.81 cm，通过；回退 3 次 | 0/2 ／ 0/2 | 25.7 %／21.4 % |
| TACO `seq_20231031_116`（4.5 s，双手） | 430 s；物体 1.78 cm，通过；回退 3 次 | 0/2 ／ 0/2 | 75.3 %／56.6 % |
| DexterHand `Cuboid_00_16200`（20 s，左手） | 1543 s；上游在第 309 帧（15.5 s）提前停止：物体丢失且回退用尽，共回退 10 次；没有 plan | 未评估，计为失败 | — |

每个 rollout 都经 `spider.h2r.check` 确认提交的控制与 plan 一致。参考导入在上述六个任务上全部通过，包括右手单手、左手单手和 DexterHand；封存自检在真实任务上的偏差为 0。

### 开环回放对物理差异的敏感度

六个任务中五个的规划在私有世界里通过了上游判定（DexterHand 一条被上游提前停止），但 bench 评估中只有一个任务成功。为区分原因，把同一份封存目标放进单个 MuJoCo world 无扰动地回放，改变版本或物理步长，比较物体位置相对规划参考的平均误差：

| 任务 | 规划世界：3.17.0、100 Hz | 3.17.0、300 Hz | 3.11、300 Hz（评估的版本与步长） |
| --- | --- | --- | --- |
| ARCTIC `s01-box_use_01` | 1.1 cm | 13.9 cm，3.2 s 起超过 10 cm | 14.1 cm，3.3 s 起超过 10 cm |
| OakInk2 双手 | 0.8 cm | 未测 | 1.0 cm |
| OakInk2 右手 | 0.8 cm | 未测 | 1.3 cm（结束时 8.4 cm） |
| HOT3D | 2.9 cm | 37.2 cm，1.5 s 起超过 10 cm | 29.6 cm，2.6 s 起超过 10 cm |
| TACO | 1.8 cm | 2.2 cm | 2.5 cm |

另做了一次诊断：用 `--config` 把私有步长改为 300 Hz 重新规划 ARCTIC box（952 s，私有世界中 1.05 cm、通过）。这份 plan 在 3.17.0、300 Hz 下回放为 1.05 cm，在 3.11、300 Hz 下 8.1 s 起物体脱手。

结论：

- 离线回放没有反馈，plan 只对规划时的物理成立。ARCTIC box 与 HOT3D 这两个任务，只换步长或只换 MuJoCo 版本都会让物体脱手；把规划步长改成与评估相同并不够，版本差异单独就足以破坏回放。
- OakInk2 的两个任务和 TACO 在无扰动回放中可以迁移。其中只有 OakInk2 双手任务在 bench 评估中成功；另外两个的 AUC10-All 较高但未达到 SR 的判定，原因未分析。
- 因此当前配置（规划 3.17.0、100 Hz；评估 3.11、300 Hz）下，上游判定通过不代表评估成功。上游在自己场景上的通过率更不能外推。

另用 h2r_bench 方法页的完整命令序列（含 `time-stage` 构建记录与 `--build-record`）在 OakInk2 双手任务上重新走了一遍：规划 126 s，评估结果与上表逐位相同的 AUC（92.255 %），汇总给出的处理耗时为人手时长的 28.8 倍。

未运行：suite 其余任务、MJWarp 评估 backend、正式口径（10 个 seed）的评估。
