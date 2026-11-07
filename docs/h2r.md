# h2r_bench bundle 与统一评估

每个序列依次运行 `spider.h2r.prepare` 和 `spider.h2r.plan` 一次，再用同一份封存的控制序列进行 h2r_bench `sim` 10 次与 `sim-dr` 10 次评估。`prepare` 只读取 bench 分发的固定 task，生成 IK 参考；`plan` 在 SPIDER 私有 MJWarp 世界中执行原生 sampling MPC，保存整条机器人位置目标。统一 eval 默认使用 MuJoCo，客户端只按共同 tick 回放控制，不重新规划。

算法、标定、规划与回放都在本仓库；bench 持有评估世界、初始化、时钟、终止和共同指标。部署方式记作 `offline-control-replay`。此入口属于显式 H2R 任务适配，不能当作原生 ARCTIC/XHand 配方或论文性能复现。

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
h2r-bench prepare-evaluation --scene-bundle "$HUMAN" --robot-bundle "$ROBOT" \
  --physics-hz 240 --control-hz 30 --output "$TASK"
h2r-bench prepare-metrics task="$TASK" output="$SPEC"
```

然后在 SPIDER 环境准备：

```bash
python -m spider.h2r.prepare --task "$TASK" --output "$PREPARED"
```

可选 `--config recipe.yaml` 覆盖 `ik` 或 `mpc` 的字段；字段及默认值见 `spider/h2r/settings.py`，未知字段会报错。配置改变需要新的 prepared 目录。初始化碰撞检查失败时保留诊断，但不生成可部署的 `h2r_binding.json`。

| 产物 | 用途 |
| --- | --- |
| `trajectory_keypoints.npz` | 同一时间轴上的 wrist 与 fingertip targets |
| `trajectory_kinematic.npz` | 全部源帧的 `time_s/qpos/qvel/ctrl`；ctrl 按 actuator 顺序 |
| `initial_robot.json`、`initialization.json` | 具名零速度初态提案与穿透检查 |
| `resolved_config.yaml`、`calibration.json` | 实际配方与 Sharpa wrist 标定 |
| `h2r_binding.json` | task/bundle/robot 身份、方法版本及产物 hash |

prepared 只含方法参考、标定、配置与初态，可整体搬移；规划与在线诊断必须同时提供匹配的 task。
`scene.planning_model` 在每次构建前重新验证 task 的身份和文件闭包，直接在内存中添加 tracking sites。
不读取 prepared 中的任何场景，不提供 `--model-path`、机器人资产或物体物理覆盖参数。

准备和计划格式分别为 `spider-h2r-v2`、`spider-h2r-plan-v2`。旧版本因携带独立 planning scene 而不再接受，
需要从 bench task 重新运行 prepare 和 plan；不能靠修改格式字段迁移旧目录。
任务目录应由运行方以只读方式提供，评估服务固定其选择的 `task_id`。
文件 hash 校验用于内容一致性，不是来源签名，也不构成共享文件系统上的进程隔离。

在 SPIDER 环境对完整序列规划一次，规划 seed 与后续 eval seeds 分开：

```bash
CUDA_VISIBLE_DEVICES="$GPU" python -m spider.h2r.plan \
  --prepared "$PREPARED" --task "$TASK" --output "$PLAN" \
  --seed 0 --device cuda:0
```

规划复用原生 optimizer 和 `mjwp.step_env`，在私有仿真中滚动优化并执行选定控制。仅初始时设置共同物体姿态、IK 机器人初态与零速度；之后物体通过私有物理推进，不按参考逐帧重置。输出 `trajectory_mjwp.npz` 的 `ticks/time_s/joint_targets/ctrl` 是已选定控制，`qpos/qvel` 与 `planning.jsonl` 仅作规划诊断。`plan.json` 记录完整任务网格、规划 seed、配方、版本、耗时、输入身份和文件哈希；失败或中断不产生可部署的完整 manifest。

20 次 eval 必须全部引用同一个 `PLAN`，不按 eval seed 重新运行 `plan`。

在 bench 的评估环境启动服务：

```bash
h2r-bench evaluate-server task="$TASK" spec="$SPEC" mode=sim backend=mujoco \
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

- 复用 `preprocess/ik_fast.py` 的 wrist/fingertip/posture tasks 和 DAQP 求解。冻结全部非机器人 DoF，保留全部帧与物体。当前配方不启用 collision-constrained IK；初态仍接受共同碰撞检查，不能通过移动物体或放宽阈值部署。
- 首批标定为 `virtual_wrist_sharpa_wave_bimanual` 与 `virtual_wrist_sharpa_wave_right`。腕部六个标量关节经真实运动学 IK 求解；机器人关节顺序与 actuator 顺序分别映射。其他机器人在 prepare 阶段明确拒绝。
- 支持 task v1/v2/v3 的具名单/多 free root、被动 hinge 与固定环境；读取 bench 装配后展开的 MJCF，拒绝 include、外部 model/plugin 及隐式资源目录；仅接受 unit-gear、无执行器内部动态的独立位置 actuator。遗漏动态关节、额外 actuator、缺失观测、身份或时钟不匹配会报错。
- 默认开发配方为 128 样本、2 iterations、0.4 s horizon、0.1 s knot、8 个物理子步。动作周期与 task 的共同控制周期相同，每个目标在全部子步保持；仅私有规划过程逐步重规划，eval 复用固定控制。horizon/knot 必须落在共同控制网格上。
- 原生 sampling optimizer、加权 qpos 范数、qvel 项及 terminal multiplier 保持不变；按名字配置 wrist/finger/object 权重与噪声，被动 hinge 使用 object rotation 权重。所有动态物体都参与私有跟踪，包括 context 物体。
- 关闭 contact guidance/reward；`num_dr=1`，规划使用 nominal 参数。保留共同模型的质量、几何、弹簧、重力补偿与伺服；只按配置改变私有积分 timestep，不继承原生 hand 模式的 solver/contact 重写。评估 `sim-dr` 的扰动留在 bench，离线控制不对观测扰动作 MPC 修正。
- 以真实 tick 采样未来参考，freejoint quaternion 用 SLERP，速度在相同网格上做流形差分。末尾 hold/零速度仅供私有 horizon，bench 的终点不变。
- 同一序列只有一个规划 seed 与控制序列；eval seed 不改变规划 RNG 或控制。终止 episode 不再发动作。文件身份、控制网格不完整或非有限目标在连接 eval 前拒绝。

通信协议为 `h2r-evaluation-v1`：4-byte 大端长度 + JSON，`hello/start/frames/step`。仅返回 `joint_targets`，没有 root wrench、物体辅助力、评估 reset 或方法提前终止。每次 eval 的 binding 与控制日志记录相同 `plan_id`，规划耗时与 eval 耗时分别保存。

## 检查与已验证范围

```bash
python -m unittest discover -s tests -p 'test_h2r*.py' -v
```

回放测试覆盖 20 个独立 episode 的相同控制、扰动观测不影响输出、末尾不足一个控制周期、游标检查、不完整／非有限计划及文件篡改拒绝。

边界测试覆盖乱序 actuator、多 root+hinge、body 原点速度、源姿态转换、端点、封存完整性和多 episode 终止。兼容单手布局下，原生与适配 reward/terminal reward、固定随机输入的控制采样逐值一致。

历史在线 MPC 运行结果保留在 bench 的 `runs/checks/spider-h2r-20261107/` 及对应 evaluation batch 中，不代表离线控制回放结果。各数据集离线规划与物理评估的实际通过范围，以新批次运行记录为准。

## 显式在线诊断

`evaluation --prepared PREPARED --device cuda:0` 仍可显式运行在线 MPC，每个 eval episode 根据真实观测重新规划。`spider.h2r.check` 只适用于这种在线名义记录。此模式与默认的一次规划／重复回放分开记录，不能混报成绩。

## 任务与模型边界检查

测试覆盖 task 资源篡改、重新封存后的任务身份变化、prepared 添加模型／路径或伪造 hash、
包外依赖与 XML 隐式资源目录拒绝。数值检查对照原 task 编译模型，确认 tracking sites 与规划 timestep
不会改变质量惯量、关节、碰撞、mesh、执行器或共同 solver 参数。

历史 Leap wrench、JSON-lines 协议、原生数据集／场景生成及 `spider-mjwp` 入口已移除。
原生复现使用上游 checkout；已有运行记录不在本次代码迁移中改写。
