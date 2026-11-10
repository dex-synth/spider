# h2r_bench bundle 与统一评估

每个序列依次运行 `spider.h2r.prepare` 和 `spider.h2r.plan` 一次，再用同一份封存的控制序列进行 h2r_bench `sim` 10 次与 `sim-dr` 10 次评估。`prepare` 只读取 bench 分发的固定 task，生成 IK 参考；`plan` 在 SPIDER 私有 MJWarp 世界中执行原生 fast sampling MPC，保存整条机器人位置目标。统一 eval 默认使用 MuJoCo，客户端只按共同 tick 回放控制，不重新规划。

算法、标定、规划与回放都在本仓库；bench 持有评估世界、初始化、时钟、终止和共同指标。部署方式记作 `offline-control-replay`。此入口属于显式 H2R 任务适配，不能当作原生 ARCTIC/XHand 配方或论文性能复现。

与 original 的实现、参数及时间边界对照见 [上游差异说明](upstream-differences.md)。

## 环境

使用独立 Python 3.12 环境，不需要安装 h2r_bench。常规安装使用 `uv sync --extra h2r`；本机 CUDA 12 配方使用已有约束：

```bash
bash scripts/h2r_environment.sh /path/to/spider-env
```

脚本安装 `.[h2r]`（显式声明 SciPy、PyYAML、DAQP），保持 MuJoCo 3.7.0、mujoco-warp 3.7.0.1、Warp 1.12.1、Torch 2.7.1、Mink 1.1.0。完整约束见 `scripts/h2r_runtime_constraints.txt`；它是当前 CUDA 12 配方，与原生 `uv.lock` 的默认平台解析分开。已有容器可直接使用 `/opt/h2r/spider/bin/python`，无需重装。

此入口使用 bundle 中已解码的 MANO joints 与原始 root rotation，不读取原始数据集或 MANO 模型。原生 raw-data processors、自带机器人资产与场景生成入口已移除。

## 准备与部署

以下路径均由调用者提供；输入目录已封存，输出目录必须不存在。先在 bench 环境固定任务与指标：

```bash
h2r-bench prepare-evaluation method=spider scene_bundle="$HUMAN" \
  robot_bundle="$ROBOT" output="$TASK"
h2r-bench prepare-metrics task="$TASK" output="$SPEC"
```

新任务默认 300 Hz 物理、100 Hz 控制，每个原生位置目标保持 3 个物理 tick；计分仍使用共同参考时钟。
任务准备与评估都指定 `method=spider`，评估服务会核对封存任务的控制频率。

然后在 SPIDER 环境准备：

```bash
python -m spider.h2r.prepare --task "$TASK" --output "$PREPARED"
```

可选 `--config recipe.yaml` 覆盖 `ik` 或 `mpc` 的字段；字段及默认值见 `spider/h2r/settings.py`，未知字段会报错。配置改变需要新的 prepared 目录。初始化关节限位或碰撞检查失败时保留诊断，但不生成可部署的 `h2r_binding.json`。

| 产物 | 用途 |
| --- | --- |
| `trajectory_keypoints.npz` | 同一时间轴上的 wrist 与 fingertip targets |
| `trajectory_kinematic.npz` | 完整共同时钟上的参考，以及原生 crop/filter 数组与源时间；ctrl 按 actuator 顺序 |
| `trajectory_ikrollout.npz` | 上游带噪声的私有前向诊断，不作为规划参考 |
| `initial_robot.json`、`initialization.json` | 具名零速度初态提案与穿透检查 |
| `resolved_config.yaml`、`calibration.json` | 实际配方与 Sharpa wrist 标定 |
| `h2r_binding.json` | task/bundle/robot 身份、方法版本及产物 hash |

prepared 只含方法参考、标定、配置与初态，可整体搬移；规划必须同时提供匹配的 task。
`scene.planning_model` 在每次构建前重新验证 task 的身份和文件闭包，直接在内存中添加 tracking sites。
不读取 prepared 中的任何场景，不提供 `--model-path`、机器人资产或物体物理覆盖参数。

准备和计划格式分别为 `spider-h2r-v4`、`spider-h2r-plan-v4`。v1 携带独立 planning scene，v2 使用旧 fast IK／逐 tick 规划，v3 使用论文 IK／普通 MPC，均不再接受，
需要从 bench task 重新运行 prepare 和 plan；不能靠修改格式字段迁移旧目录。
新规划还要求 prepared 的 `pipeline.rotation_processing=sign-continuous-mean-nlerp-v1`。缺失或未知标记必须重新 prepare，避免旧平滑结果混入修正后的 planner。该标记随 preparation 进入 plan 身份；历史完整 v4 plan 仍可按原身份进行 CPU 回放与审计。
任务目录应由运行方以只读方式提供，评估服务固定其选择的 `task_id`。
文件 hash 校验用于内容一致性，不是来源签名，也不构成共享文件系统上的进程隔离。

在 SPIDER 环境对完整序列规划一次，规划 seed 与后续 eval seeds 分开：

```bash
CUDA_VISIBLE_DEVICES="$GPU" python -m spider.h2r.plan \
  --prepared "$PREPARED" --task "$TASK" --output "$PLAN" \
  --seed 0 --device cuda:0
```

规划复用原生 fast optimizer 和 `mjwp.step_env`。每段直接使用最佳候选的 post-step qpos/qvel 并恢复其状态快照，不重复仿真均值控制。仅初始时设置共同物体姿态、IK 机器人初态与零速度；之后物体通过私有物理推进，不按参考逐帧重置。输出 `trajectory_mjwp.npz` 的 `ticks/time_s/joint_targets/ctrl` 是已选定控制，`native_*` 保存原生网格轨迹；`source_time_s` 标记所取原生帧。`qpos/qvel` 是该控制执行后的诊断状态（时间为 `source_time_s + native_action_dt_s`），不冒充发布时刻的评估状态；`planning.jsonl` 记录优化触发与轮数。`plan.json` 记录完整任务网格、规划 seed、配方、版本、耗时、输入身份和文件哈希；失败或中断不产生可部署的完整 manifest。

20 次 eval 必须全部引用同一个 `PLAN`，不按 eval seed 重新运行 `plan`。

在 bench 的评估环境启动服务：

```bash
h2r-bench evaluate-server method=spider task="$TASK" spec="$SPEC" mode=sim backend=mujoco \
  num_trajectories=10 'seeds=[0,1,2,3,4,5,6,7,8,9]' socket="$SOCKET" output="$ROLLOUT" \
  timeout_s=900 video.enabled=false
```

待日志出现 `Evaluation ready:` 后，在 SPIDER 环境连接：

```bash
CUDA_VISIBLE_DEVICES= python -m spider.h2r.evaluation \
  --plan "$PLAN" --task "$TASK" --socket "$SOCKET" \
  --device cpu --output "$CONTROLLER_OUTPUT" --timeout-s 900
```

对 `mode=sim-dr` 再启动独立的 server/socket/输出目录，客户端继续使用相同 `PLAN`。回放无需 CUDA，不读取评估状态改变控制；每个 episode 独立检查时钟并推进游标。MuJoCo 实际施加机器人位置目标，并保留共同执行器反馈、扰动和终止规则；不将规划状态写入评估世界。共同计分在 bench 环境执行：

```bash
h2r-bench score-evaluation --run "$ROLLOUT" --spec "$SPEC" --output "$SCORES"
h2r-bench summarize-evaluations --batch "$BATCH"
```

批次目录、预定集合与完整命令见 h2r_bench 的 `docs/methods.md`。完成时间轴不等于操作成功，最终成绩读取共同指标。

## 适配配方与边界

固定上游 `44717007de41cbef7565dff7ff9f4453557a2d3d`，使用 README 推荐的 `preprocess/ik_fast.py` → `examples/run_mjwp_fast.py`。固定 YAML 保存在 `spider/h2r/config/`；配置只读取算法字段，任务、机器人、资产与速度仍由 bench 提供。显式 `--config` 最后覆盖，实际值封存到 `resolved_config.yaml`。

| 数据集 | override | samples／iterations | ctrl／horizon／knot（s） | temperature | sanity check |
| --- | --- | --- | --- | --- | --- |
| HOT3D／无专用配方 | 基础 default | 1024／32 | .4／1.6／.4 | .1 | 3 s |
| ARCTIC | arcticv2_fast | 1024／4 | 1.5／3／1 | 1 | 关闭 |
| OakInk2 | oakinkv2_fast | 1024／4 | 1.5／3／1 | 1 | 3 s |
| GigaHand | gigahand_fast | 1024／8 | 1.5／3／.5 | 1 | 3 s |
| OakInk | oakink_fast | 1024／8 | 1.5／3／1 | 1 | 3 s |

噪声、reward、误差门槛、容量等也随对应 YAML 合并，不混搭配方。基础容量为 120 contacts／350 constraint rows，Arctic 为 256／768；溢出按失败记录，不静默增大。fast 不使用 improvement 提前停止，每个 attempt 跑完配置轮数；失败最多 4 个 attempt，随后 runner 执行最多 3 次 local retry，iterations 每级乘 2、noise 每级乘 sqrt(2)。恢复失败时回到同一锚点，按上游上限退出。失败后保留原生截断／最后候选诊断，不生成可部署 plan，也不缩短共同计分分母。

- **IK**：Mink/DAQP，先 wrist-only 200 步，再加入 fingertip 300 步；每帧执行 round(ref_dt/.005) 步，每步仍为 .005 s。物体 qpos 每步直接写入方法参考。保留原生成本与 damping，不加碰撞避免、速度限制或新腕算法。
- **参考处理**：使用数据集的真实 reference rate，保留默认最后一帧裁剪、3 帧 valid 平均、差分、首帧丢弃。free-root 四元数在平滑前归一化并对齐符号，平滑后重新归一化，再由 `mj_differentiatePos` 求 qvel。三处 qpos 插值保持原位置、时间映射及分量线性插值，前后处理 quaternion slices，形成 NLERP；不使用 SLERP。平移、scalar joints 和 ctrl 保持原值；qvel 独立线性插值并保持端点速度，不作为最终 NLERP 曲线的解析导数。非法源四元数或近零平均结果明确拒绝。原始 IK 与 filtered 数组均保存。
- **参考时间边界**：上游 `process_config` 先从基础 .02 s 求 ref_steps，再读数据集 ref_dt；此顺序保留在插值因子中。上游 H*n 插值数组会重置起点并改变时间跨度；这里将每个插值值明确对应到原始窗口中心的源时间，再采样到私有 sim_dt 网格。完整任务的前后端点保持，不裁剪共同任务或改变其速度。这是显式时间边界转换，不能声称与上游错误计时逐帧等价。
- **规划**：最后一轮使用真实最佳样本；优化参考为 `[s+1:s+horizon+1]`，初始化控制为 `[0:horizon]`。状态快照推进、移窗、重试、boost 与锚点回退均发生在私有世界。普通成功不会自动清除 local boost；越过恢复锚点后才按上游重置。
- **初始化与物理**：方法平滑参考与共同物体初态分开。共同初态使用原始物体姿态、IK 机器人位置、零速度、无 warm-up。上游 sanity check 在私有 CPU 模型以 .002 s 积分检查最大 root 漂移，默认 3 s／.01 m／.3 rad。评估场景、MuJoCo backend 和初始化规则不变。
- **控制时钟**：私有规划保持 100 Hz，重规划间隔按数据集配方。新任务以 `method=spider` 选择 300 Hz 物理／100 Hz 发布。历史 HOT3D task 的 240 Hz 物理／30 Hz 发布只用于复查旧结果：零阶保持会丢弃部分原生动作变化，不能声称完整执行了 100 Hz 控制。显式重建历史条件时，在任务准备和评估两端均声明 `method=spider method.control_hz=30`，准备时另设 `physics_hz=240`；新频率任务须重新 prepare／plan，不能修改旧 task 或 plan。
- **具名映射**：Sharpa 双手／右手虚拟腕、task v1/v2/v3、多 free root 和被动 hinge。reward/noise 的 wrist/finger/root 地址由 task 解析，被动 hinge 使用 object rotation 权重；成功门槛保留原生 root 检查。没有资产 fallback，不删除 context 物体。
- **明确例外**：按用户要求，`num_dr=num_dyn=1` 使用 nominal 参数，不保留上游 `linspace(...,1)` 产生的非零偏移；free-root 旋转使用上述符号对齐与归一化修正，不能宣称与 original 逐值等价。其他 DR 数量明确拒绝。contact guidance/Gibbs 在上游 fast 不受支持；保持关闭。不顺带修正上游 NaN 选择等其他缺陷。
- **部署**：每个 seq 离线规划一次，sim/sim-dr 复用同一控制。v4 不支持 `--prepared` 在线入口，因为已执行的真实评估动作无法回退；不能将删掉回退的循环称为完整 upstream fast。

通信协议为 `h2r-evaluation-v1`：4-byte 大端长度 + JSON，`hello/start/frames/step`。仅返回 `joint_targets`，没有 root wrench、物体辅助力、评估 reset 或方法提前终止。每次 eval 的 binding 与控制日志记录相同 `plan_id`，规划耗时与 eval 耗时分别保存。

## 检查与已验证范围

评估结束后，在 SPIDER 环境用 CPU 检查 v4 plan 与 bench 实际接收、记录的控制：

```bash
python -m spider.h2r.check --plan "$PLAN" --task "$TASK" \
  --run "$ROLLOUT" --output "$REPLAY_CHECK"
```

默认核对所有 seeds；`--episode seed-0` 可只查一个。检查 task／plan 身份、具名初态、完整 plan 时间网格、逐次提交控制和 physical 记录中的 `submitted_target`，同时核对 JSONL／NPZ 的控制记录。适用于 `sim` 和 `sim-dr`，不要求受延迟、限幅影响的 `effective_target` 等于 plan，也不把规划状态当成实际评估状态。

提前失败仍检查已记录前缀，并报告 `reason`、`coverage` 和 `complete=false`；`control_consistent=true` 只说明该前缀控制一致，完整执行须同时检查 `full_time_coverage` 与 `complete`，操作成功仍由 bench 指标判断。文件缺失、控制不一致或虚报完整时域会使检查报错。旧在线诊断参数 `--prepared/--controller-log/--device/--steps` 已移除。

```bash
python -m unittest discover -s tests -p 'test_h2r*.py' -v
```

回放测试覆盖 20 个独立 episode 的相同控制、扰动观测不影响输出、末尾不足一个控制周期、游标检查、不完整／非有限计划及文件篡改拒绝。

边界测试覆盖乱序 actuator、多 root+hinge、body 原点速度、源姿态转换、封存完整性和多 episode 终止。旋转测试覆盖符号等价、跨 180°、三处 NLERP、非法输入与近零平均拒绝；fast 行为测试覆盖数据集配置、标量过滤与速度保持、固定轮数／4 次 attempt、最佳样本选择、快照推进、控制移窗、local boost、锚点恢复、失败截断和非整比发布。

源码数值对照与真实任务验证的结果保存在 bench 对应 `runs/checks/`；单元测试通过不代表全序列成功或论文性能复现。历史在线／论文 IK 产物按各自运行记录解释，不混用为新版 fast 结果。

## 任务与模型边界检查

测试覆盖 task 资源篡改、重新封存后的任务身份变化、prepared 添加模型／路径或伪造 hash、
包外依赖与 XML 隐式资源目录拒绝。数值检查对照原 task 编译模型，确认 tracking sites 与规划 timestep
不会改变质量惯量、关节、碰撞、mesh、执行器或共同 solver 参数。

历史 Leap wrench、JSON-lines 协议、原生数据集／场景生成及 `spider-mjwp` 入口已移除。
原生复现使用上游 checkout；已有运行记录不在本次代码迁移中改写。
