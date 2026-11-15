# h2r_bench 任务与统一评估

每个序列依次运行 `spider.h2r.prepare` 和 `spider.h2r.plan` 一次，再用同一份封存的控制序列进行 h2r_bench 的 `sim` 与 `sim-dr` 评估。`prepare` 只读取 bench 分发的 task 目录，生成 IK 参考；`plan` 在 SPIDER 私有 MJWarp 世界中执行原生 fast sampling MPC，保存整条机器人位置目标。统一评估默认使用 MuJoCo，客户端按自己的 0.01 s 控制周期回放控制，不重新规划。

算法、标定、规划与回放都在本仓库；bench 持有评估世界、物理步长、初始化准入、终止和共同指标。部署方式记作 `offline-control-replay`。此入口属于显式 H2R 任务适配，不能当作原生 ARCTIC/XHand 配方或论文性能复现。

与 original 的实现、参数及时间边界对照见 [上游差异说明](upstream-differences.md)。

## 环境

使用独立 Python 3.12 环境，不需要安装 h2r_bench。常规安装使用 `uv sync --extra h2r`；本机 CUDA 12 配方使用已有约束：

```bash
bash scripts/h2r_environment.sh /path/to/spider-env
```

脚本安装 `.[h2r]`（显式声明 SciPy、PyYAML、DAQP），保持 MuJoCo 3.7.0、mujoco-warp 3.7.0.1、Warp 1.12.1、Torch 2.7.1、Mink 1.1.0。完整约束见 `scripts/h2r_runtime_constraints.txt`；它是当前 CUDA 12 配方，与原生 `uv.lock` 的默认平台解析分开。已有容器可直接使用 `/opt/h2r/spider/bin/python`，无需重装。

## 输入

唯一的输入是 bench 的 **task 目录**（`h2r-task-v1`：`task.json`、`scene.xml`、`compiled.json`，以及 task 内或所属 suite 共享的 demo 与 robot 包）。四个数据集（ARCTIC、OakInk2、HOT3D、TACO）的 task 结构相同，没有按数据集区分的读取代码。

task 由 `spider/h2r/h2r_io/` 读取。它是 bench 加载器的副本（`h2r-bench export-loader` 生成，`SOURCE.json` 记录来源 commit 与文件 hash），随方法源码一起进入产物的版本身份；不在本仓库修改，Ruff 不检查该目录。bench 更新加载器后重新导出，并用 `h2r-bench check-loader spider/h2r/h2r_io` 核对。加载时校验 task、demo、robot 的全部文件及场景资源闭包，被改动的输入在编译模型前被拒绝。

方法使用的内容：

| 来源 | 用途 |
| --- | --- |
| `task.scene_xml` | IK 与 MPC 的私有模型；只在内存中添加无质量 tracking sites |
| `task.robot.name`、`task.hand_sides` | 选择手部标定；只处理参与任务的手，单手与双手走同一代码路径 |
| `task.robot.semantics` | 关节名称、单位和 wrist／finger 分组 |
| `task.object_binding` | 自由根、部件、物体关节的 MJCF 名称与参考数组名 |
| `task.demo` | `joints_m` 给出腕部位置与五个指尖，`mano_quat[:, 0]` 是腕部在基准世界系中的旋转；物体根位姿与关节角参考 |
| `task.dataset` | 选择数据集配方；`task.duration_s` 决定规划长度 |

不读取原始数据集或 MANO 模型，不重新解码 MANO。task 不含控制频率、评估物理步长或初始穿透阈值；这些分别属于方法与评估。

## 准备与部署

以下路径均由调用者提供；输出目录必须不存在。在 SPIDER 环境准备：

```bash
python -m spider.h2r.prepare --task "$TASK" --output "$PREPARED"
```

可选 `--config recipe.yaml` 覆盖 `ik` 或 `mpc` 的字段；字段及默认值见 `spider/h2r/settings.py`，未知字段会报错。配置改变需要新的 prepared 目录。

prepare 对 IK 得到的零速度初态做 SPIDER 自己的预检：关节限位，以及任一接触穿透不超过 `PRECHECK_MAX_PENETRATION_M = 0.02 m`（`spider/h2r/prepare.py`，与评估的默认准入阈值相同）。原生 IK 没有碰撞避免，示范中手贴近桌面或物体的首帧可能穿透；这是方法本身的缺陷，计为该任务的失败，不修正初态，也不把任务移出集合。预检不通过时 prepare 照常完成，`initialization.json` 记录穿透的几何体对；plan 不运行 MPC，只封存这个初态（`deployment: initial-pose-only`，不含控制）。评估客户端照常提出该初态，由评估的初始化准入记录为 `initial_collision`；若评估在更宽松的阈值下接受了它，客户端在第一个控制周期明确报错，不发送任何控制。

| 产物 | 用途 |
| --- | --- |
| `trajectory_keypoints.npz` | 同一时间轴上的 wrist 与 fingertip targets |
| `trajectory_kinematic.npz` | 完整参考时钟上的参考，以及原生 crop/filter 数组与源时间；ctrl 按 actuator 顺序 |
| `trajectory_ikrollout.npz` | 上游带噪声的私有前向诊断，不作为规划参考 |
| `initial_robot.json`、`initialization.json` | 具名零速度初态提案与方法侧预检结果 |
| `resolved_config.yaml`、`calibration.json` | 实际配方与手部标定 |
| `h2r_binding.json` | `task_id`、`demo_id`、`robot_id`，数据集、参与手、方法版本及产物 hash |

prepared 只含方法参考、标定、配置与初态，可整体搬移；规划必须同时提供匹配的 task。
不读取 prepared 中的任何场景，不提供 `--model-path`、机器人资产或物体物理覆盖参数。

准备和计划格式分别为 `spider-h2r-v5`、`spider-h2r-plan-v5`，绑定 `task_id`、`demo_id` 与 `robot_id`。v4 及更早格式面向此前的 task 目录与协议，一律拒绝并提示从 `--task` 重新生成；不能靠修改格式字段迁移旧目录。prepared 还须带 `pipeline.rotation_processing=sign-continuous-mean-nlerp-v1`，该标记随 preparation 进入 plan 身份。
文件 hash 校验用于内容一致性，不是来源签名，也不构成共享文件系统上的进程隔离。

在 SPIDER 环境对完整序列规划一次，规划 seed 与后续评估 seeds 分开：

```bash
CUDA_VISIBLE_DEVICES="$GPU" python -m spider.h2r.plan \
  --prepared "$PREPARED" --task "$TASK" --output "$PLAN" \
  --seed 0 --device cuda:0
```

规划复用原生 fast optimizer 和 `mjwp.step_env`。每段直接使用最佳候选的 post-step qpos/qvel 并恢复其状态快照，不重复仿真均值控制。仅初始时设置共同物体姿态、IK 机器人初态与零速度；之后物体通过私有物理推进，不按参考逐帧重置。

plan 发布在 SPIDER 的原生动作时钟上：`plan.json` 的 `control_dt_s` 为 `mpc.sim_dt`（默认 0.01 s），`controls = ceil(task.duration_s / control_dt_s)`。`trajectory_mjwp.npz` 的 `time_s = arange(controls) * control_dt_s`、`joint_targets` 和 `ctrl` 是已选定控制；`qpos/qvel` 是各控制执行后的私有诊断状态（时间为 `state_time_s = time_s + control_dt_s`），不冒充评估状态。plan 不保存物理 tick、评估步长或每周期步数。`planning.jsonl` 记录优化触发与轮数；`plan.json` 记录规划 seed、配方、版本、耗时、输入身份和文件哈希。失败或中断不产生可部署的 manifest：原生不可恢复失败写 `failure.json` 与 `trajectory_failed.npz`，其他错误写 `error.json`。

所有评估必须引用同一个 `PLAN`，不按评估 seed 重新运行 `plan`。

在 bench 的评估环境固定指标、声明并启动服务（完整流程见 h2r_bench 的 `docs/methods/spider.md`）：

```bash
h2r-bench prepare-metrics task="$TASK" batch="$BATCH" name=demo
h2r-bench declare-evaluation --batch "$BATCH" --method spider --case demo --task "$TASK" \
  --spec inputs/metric-specs/demo --run-prefix demo-spider --backend mujoco \
  --training-backend mjwarp --variant offline-control-replay
h2r-bench evaluate-server task="$TASK" spec="$BATCH/inputs/metric-specs/demo" mode=sim backend=mujoco \
  socket="$SOCKET" batch="$BATCH" name=demo-spider-sim training_backend=mjwarp video.enabled=false
```

socket 出现后，在 SPIDER 环境连接：

```bash
CUDA_VISIBLE_DEVICES= python -m spider.h2r.evaluation \
  --plan "$PLAN" --task "$TASK" --socket "$SOCKET" --output "$CONTROLLER_OUTPUT"
```

对 `mode=sim-dr` 再启动独立的 server/socket/输出目录，客户端继续使用相同 `PLAN`。回放无需 CUDA，不读取评估状态改变控制，因此 start 不请求任何 body 或接触观测；每个 episode 独立检查时钟并推进游标。MuJoCo 实际施加机器人位置目标，并保留共同执行器反馈、扰动和终止规则；不将规划状态写入评估世界。共同计分在 bench 环境执行：

```bash
h2r-bench summarize-evaluations --batch "$BATCH"
```

完成时间轴不等于操作成功，最终成绩读取共同指标。

## 适配配方与边界

固定上游 `44717007de41cbef7565dff7ff9f4453557a2d3d`，使用 README 推荐的 `preprocess/ik_fast.py` → `examples/run_mjwp_fast.py`。固定 YAML 保存在 `spider/h2r/config/`；配置只读取算法字段，任务、机器人、资产与速度仍由 bench 提供。解析顺序为上游基础默认值、数据集 fast override、H2R 适配层 `spider/h2r/config/h2r.yaml`、显式 `--config`，实际值封存到 `resolved_config.yaml`。

配方按 `task.dataset` 选择，`settings.DATASET_RECIPES` 恰好列出 bench 的四个数据集；未列出的数据集直接报错，不回退到其他配方。

| 数据集 | override | samples／iterations | ctrl／horizon／knot（s） | temperature | sanity check |
| --- | --- | --- | --- | --- | --- |
| ARCTIC | `arcticv2_fast` | 1024／4 | 1.5／3／1 | 1 | 关闭（上游配方即关闭） |
| OakInk2 | `oakinkv2_fast` | 1024／4 | 1.5／3／1 | 1 | 关闭（上游 3 s） |
| HOT3D | 无（基础 default） | 1024／32 | .4／1.6／.4 | .1 | 关闭（上游 3 s） |
| TACO | 无（基础 default） | 1024／32 | .4／1.6／.4 | .1 | 关闭（上游 3 s） |

上游没有 HOT3D 和 TACO 的 fast override，二者使用固定的基础默认值；不借用其他数据集的配方。上游另有 GigaHand、OakInk 的 override，bench 不分发这两个数据集，其副本已从本仓库移除，需要时取自上述上游 revision。

噪声、reward、误差门槛、容量等也随对应 YAML 合并，不混搭配方。基础容量为 120 contacts／350 constraint rows，Arctic 为 256／768；溢出按失败记录，不静默增大。fast 不使用 improvement 提前停止，每个 attempt 跑完配置轮数；失败最多 4 个 attempt，随后 runner 执行最多 3 次 local retry，iterations 每级乘 2、noise 每级乘 sqrt(2)。恢复失败时回到同一锚点，按上游上限退出。失败后保留原生截断／最后候选诊断，不生成可部署 plan，也不缩短共同计分分母。

- **IK**：Mink/DAQP，先 wrist-only 200 步，再加入 fingertip 300 步；每帧执行 round(ref_dt/.005) 步，每步仍为 .005 s。物体 qpos 每步直接写入方法参考。保留原生成本与 damping，不加碰撞避免、速度限制或新腕算法。
- **手部目标**：腕部旋转为 `mano_quat[:, 0]`（已在基准世界系）右乘标定的 `mano_from_wrist_rotation`；腕部位置与指尖取自 `joints_m`。标定按 `task.robot.name` 选择（当前只有 `sharpa_wave`，见 `spider/h2r/config/sharpa_wave.json`），按 `task.hand_sides` 取左、右手；其他机器人明确拒绝。
- **参考处理**：使用 demo 的真实 reference rate，保留默认最后一帧裁剪、3 帧 valid 平均、差分、首帧丢弃。free-root 四元数在平滑前归一化并对齐符号，平滑后重新归一化，再由 `mj_differentiatePos` 求 qvel。三处 qpos 插值保持原位置、时间映射及分量线性插值，前后处理 quaternion slices，形成 NLERP；不使用 SLERP。平移、scalar joints 和 ctrl 保持原值；qvel 独立线性插值并保持端点速度，不作为最终 NLERP 曲线的解析导数。非法源四元数或近零平均结果明确拒绝。原始 IK 与 filtered 数组均保存。
- **参考时间边界**：上游 `process_config` 先从基础 .02 s 求 ref_steps，再读数据集 ref_dt；此顺序保留在插值因子中。上游 H*n 插值数组会重置起点并改变时间跨度；这里将每个插值值明确对应到原始窗口中心的源时间，再采样到私有 sim_dt 网格。完整任务的前后端点保持，不裁剪共同任务或改变其速度。这是显式时间边界转换，不能声称与上游错误计时逐帧等价。
- **规划**：最后一轮使用真实最佳样本；优化参考为 `[s+1:s+horizon+1]`，初始化控制为 `[0:horizon]`。状态快照推进、移窗、重试、boost 与锚点回退均发生在私有世界。普通成功不会自动清除 local boost；越过恢复锚点后才按上游重置。
- **初始化与物理**：方法平滑参考与共同物体初态分开。共同初态使用参考首帧的物体姿态、IK 机器人位置、零速度、无 warm-up。上游 sanity check 在私有 CPU 模型以 .002 s 积分、保持首个控制，检查最大 root 漂移（上游 3 s／.01 m／.3 rad，基础默认值与 OakInk2 配方启用）；H2R 默认关闭它，见下方明确例外。评估场景、MuJoCo backend 和初始化规则不变。
- **时钟**：三个时钟各自独立。私有规划物理与动作均为 `mpc.sim_dt = 0.01 s`，重规划间隔 `ctrl_dt` 按数据集配方；plan 以 0.01 s 的动作周期发布。评估物理步长是评估设置（当前所有任务 300 Hz），客户端只从 `hello.dt_s` 得知：start 声明 `control_dt_s = 0.01`，每个动作保持 `control_dt_s / hello.dt_s` 个物理步（300 Hz 时为 3），最后一个动作可以更短。0.01 s 不是评估步长整数倍时 start 被拒绝，不取整。
- **具名映射**：多 free root、固定根和被动 hinge 均按 `task.object_binding` 的名称解析。reward/noise 的 wrist/finger/root 地址由 task 解析，被动 hinge 使用 object rotation 权重；成功门槛保留原生 root 检查，双手任务保留上游“参考全零即物体缺席”的哨兵，单手任务与上游单手分支一样没有该哨兵。没有资产 fallback，不删除 context 物体。
- **明确例外**：上游 sanity check 默认关闭：`spider/h2r/config/h2r.yaml` 设 `mpc.sanity_check_seconds: 0.0`（项目决定，2026-11-15），因为部分任务首帧物体由手握持、悬空起始，保持首个控制时物体不会静止；`--config` 设 `mpc.sanity_check_seconds: 3.0` 可恢复上游行为。按用户要求，`num_dr=num_dyn=1` 使用 nominal 参数，不保留上游 `linspace(...,1)` 产生的非零偏移；free-root 旋转使用上述符号对齐与归一化修正，不能宣称与 original 逐值等价。其他 DR 数量明确拒绝。contact guidance/Gibbs 在上游 fast 不受支持；保持关闭。不顺带修正上游 NaN 选择等其他缺陷。
- **部署**：每个 seq 离线规划一次，sim/sim-dr 复用同一控制。没有在线执行入口：已执行的真实评估动作无法回退，不能将删掉回退的循环称为完整 upstream fast。

通信协议为 `h2r-evaluation-v2`，收发与 start 请求使用 `h2r_io.client`。客户端核对协议、`task_id`、关节名称与单位；`task_id` 已确定评估方运行的就是规划所用的任务，不再逐项比较物体、观测与环境描述。start 的 binding 为 `{demo_id, robot_id, method: "spider", controller: plan.json}`。仅返回 `joint_targets`，没有 root wrench、物体辅助力、评估 reset 或方法提前终止。每次评估的 binding 与控制日志记录相同 `plan_id`，规划耗时与评估耗时分别保存。

## 检查与已验证范围

评估结束后，在 SPIDER 环境用 CPU 检查 plan 与 bench 实际接收、记录的控制：

```bash
python -m spider.h2r.check --plan "$PLAN" --task "$TASK" \
  --run "$ROLLOUT" --output "$REPLAY_CHECK"
```

默认核对所有 seeds；`--episode seed-0` 可只查一个。物理步长、每周期步数和终点取自 rollout 的 `run.json`（`h2r-rollout-v2` 的 `dt_s`、`control_ticks`、`end_tick`），不取自 task：检查终点等于 `task.duration_s`、`control_ticks * dt_s` 等于 plan 的 `control_dt_s`，再核对 task／plan 身份、具名初态、`events.jsonl` 中逐次提交的控制及其 tick。物理记录只在共同记录网格上保存状态，不含提交目标；检查其 tick 是记录网格的前缀、首帧机器人位置等于初态提案，以及 JSONL／NPZ 一致。适用于 `sim` 和 `sim-dr`，不把规划状态当成实际评估状态。

提前失败仍检查已记录前缀，并报告 `reason`、`coverage` 和 `complete=false`；`control_consistent=true` 只说明该前缀控制一致，完整执行须同时检查 `full_time_coverage` 与 `complete`，操作成功仍由 bench 指标判断。文件缺失、控制不一致或虚报完整时域会使检查报错。

```bash
python -m unittest discover -s tests -p 'test_h2r*.py' -v
```

测试用加载器的封存函数生成小型 demo／robot／task 包，或在内存中构造 `h2r_io.Task`。task 测试覆盖 tracking sites 与私有 timestep 不改变物理参数、被改动或旧格式的 task 被加载器拒绝、prepared 只含方法参考以及旧产物格式拒绝；回放测试覆盖 plan 导出的原生时钟、控制数的浮点取整、不同评估步长下的 tick 推导、游标检查、不完整／非有限计划及文件篡改拒绝；检查测试覆盖 run 自带时钟、提前失败前缀与虚报完整时域。边界测试覆盖乱序 actuator、多 root+hinge、右手／左手／双手的手部目标与标定选择、start 请求和多 episode 终止。旋转测试覆盖符号等价、跨 180°、三处 NLERP、非法输入与近零平均拒绝；fast 行为测试覆盖四个数据集的配方与未知数据集拒绝、标量过滤与速度保持、固定轮数／4 次 attempt、最佳样本选择、快照推进、控制移窗、local boost、锚点恢复和失败截断。

单元测试通过不代表全序列成功或论文性能复现。此前 task 格式下的规划与评估结果（包括 v4 产物）不延续到当前格式。

### 当前格式下的真实任务检查

2026-11-15 在 40 个任务的 suite `h2r-v1-00df5729`（Sharpa Wave）上做了链路检查，记录位于 bench 的 `runs/checks/20261115-unified-format/stage5/spider/`。检查只说明链路在当前输入与协议下可执行，不是方法成绩：规划 seed 0，评估为 MuJoCo `sim`、seeds 0 和 1、300 Hz 物理、0.01 s 控制周期。

| 任务 | 规划 | 评估结果 |
| --- | --- | --- |
| OakInk2 右手单手（夹子，带 hinge） | 原生配方，103 s | 两个 seed 均 `target_end`，734 个控制全部核对 |
| OakInk2 双手、两个刚体 | 原生配方，122 s | 两个 seed 均 `target_end` |
| ARCTIC `s01-box_use_01`（29.6 s） | 原生配方在 22 min 时人工停止：15.0–16.5 s 处的重规划未通过原生跟踪门槛，正在逐级 boost 重试。链路检查改用关闭跟踪门槛与重试的显式调试配置，444 s | 两个 seed 均在 18.97 s 以 `object_deviation` 结束；已执行前缀的 1897 个控制核对一致 |
| HOT3D `P0001_a9d6c83d_seg017` | 原生配方（基础默认值）第一个重规划步用 4×32 轮、378 s 仍未通过跟踪门槛，人工停止。链路检查改用 8 轮并关闭跟踪门槛与重试的显式调试配置，248 s | 两个 seed 均 `target_end` |
| TACO `seq_20230930_121` | prepare 被方法侧预检拒绝：IK 初态有 8 对手—桌面接触穿透 20–25 mm | 未运行 |

调试配置只改 `mpc.object_pos_threshold`、`mpc.object_rot_threshold`、`mpc.max_local_retries`、`mpc.max_revert_depth`（HOT3D 另改 `mpc.max_num_iterations`），经 `--config` 显式传入；它们的 plan 不代表原生配方的结果。另用同一调试配置在一条通过预检的 TACO 任务（`seq_20231031_116`）上走通了规划与评估。

对整个 suite 只运行 prepare 与上游 sanity check 的 CPU 诊断：40 个任务中 12 个被初态预检拒绝（TACO 7、OakInk2 4、HOT3D 1，其中 11 个是手—桌面穿透），其余 28 个中有 3 个不满足 sanity check（OakInk2、HOT3D、TACO 各 1 个）。这些是保留的原生限制，对应任务没有可部署的 plan。

右手任务的 plan 另在 600 Hz 评估步长（每控制 6 个物理步）和 `sim-dr` 下完成回放与核验；240 Hz 步长下 0.01 s 不是整数倍，客户端在 start 前报错。未运行的部分：其余 suite 任务的规划与评估、左手单手任务的真实数据（suite 中没有）、MJWarp 评估 backend、ARCTIC／HOT3D 原生配方的完整规划。

历史 Leap wrench、JSON-lines 协议、原生数据集／场景生成及 `spider-mjwp` 入口已移除。
原生复现使用上游 checkout。
