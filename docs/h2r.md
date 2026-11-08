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

可选 `--config recipe.yaml` 覆盖 `ik` 或 `mpc` 的字段；字段及默认值见 `spider/h2r/settings.py`，未知字段会报错。配置改变需要新的 prepared 目录。初始化关节限位或碰撞检查失败时保留诊断，但不生成可部署的 `h2r_binding.json`。

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

准备和计划格式分别为 `spider-h2r-v3`、`spider-h2r-plan-v3`。v1 携带独立 planning scene，v2 使用旧 fast IK／逐 tick 规划，均不再接受，
需要从 bench task 重新运行 prepare 和 plan；不能靠修改格式字段迁移旧目录。
任务目录应由运行方以只读方式提供，评估服务固定其选择的 `task_id`。
文件 hash 校验用于内容一致性，不是来源签名，也不构成共享文件系统上的进程隔离。

在 SPIDER 环境对完整序列规划一次，规划 seed 与后续 eval seeds 分开：

```bash
CUDA_VISIBLE_DEVICES="$GPU" python -m spider.h2r.plan \
  --prepared "$PREPARED" --task "$TASK" --output "$PLAN" \
  --seed 0 --device cuda:0
```

规划复用原生 optimizer 和 `mjwp.step_env`，在私有仿真中滚动优化并执行选定控制。仅初始时设置共同物体姿态、IK 机器人初态与零速度；之后物体通过私有物理推进，不按参考逐帧重置。输出 `trajectory_mjwp.npz` 的 `ticks/time_s/joint_targets/ctrl` 是已选定控制，`native_*` 保存原生网格轨迹；`source_time_s` 标记所取原生帧。`qpos/qvel` 是该原生帧的诊断状态，不冒充发布时刻的评估状态；`planning.jsonl` 记录优化触发与轮数。`plan.json` 记录完整任务网格、规划 seed、配方、版本、耗时、输入身份和文件哈希；失败或中断不产生可部署的完整 manifest。

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

固定上游 `44717007de41cbef7565dff7ff9f4453557a2d3d`，采用论文 `preprocess/ik.py` → 普通 `examples/run_mjwp.py` 路线。配置来自 `examples/config/default.yaml`，torch.compile 来自 `Config` 默认；不混入 fast optimizer、best-sample、retry 或 revert。该 revision 没有 HOT3D `_origin` 配方，因此这里是“论文入口 + 官方通用配方的共同任务适配”，不是 HOT3D 官方成绩复现。

| 时钟／预算 | 默认 |
| --- | --- |
| IK 积分／目标采样 | 200 Hz／50 Hz |
| MPC 控制采样／私有积分 | 100 Hz／100 Hz |
| MPC 重规划／horizon／knot | 0.4 s／1.6 s／0.4 s |
| 样本／最多迭代／提前停止 | 1024／32／improvement < 0.01，patience=1 |
| temperature／torch.compile | 0.1／开启 |
| 共同评估积分／控制发布 | 按 task；既有 HOT3D 为 240 Hz／30 Hz |

私有积分保留与原生 100 Hz 动作网格精确兼容的 100 Hz，不将新接入的一般 120 Hz 默认强套到该网格。每次优化后执行 40 个原生控制，再移窗 40 步；优化参考从下一步开始。零阶保持将原生控制采样到共同发布时刻，覆盖全部任务 ticks，不改变任务速度或终点。30 Hz 发布不能执行全部 100 Hz 控制变化，这是明确的部署边界差异，不声称与原生 100 Hz 执行轨迹相同。

- 论文 IK 保留 mocap wrist/object weld、finger connect、8 次随机初值与物理稳定性选择、逐帧求解、3 帧 valid 平均及差分后首帧丢弃。不增加碰撞避免、速度约束或新腕姿态算法。IK RNG 独立固定，默认 seed=0；原 CLI 未固定该 seed。
- 共同输入插值到 50 Hz 目标网格，最后不足一帧时只在私有参考中保持端点。平滑输出按窗口中心时间定位，再对应差分后的下一帧；前后端点保持覆盖完整共同任务。原生未平滑结果、有效过滤结果及时间均存入 `trajectory_kinematic.npz`。MPC 直接采样有效过滤结果，避免先降到共同网格再升采样。
- 原生读取器按数组长度插值并重新从零计时；这里按物理时间插值机器人标量位置及原生差分速度，物体 freejoint 用 SLERP 并保持共同目标。端点保持区速度为零。共同物体目标不被 IK 结果或算术四元数平滑覆盖，被动 hinge 参考也原样保留。
- 初态由 IK 提议，评估仍为零速度、无额外 warm-up；关节限位／穿透准入失败会保存 `initialization.json`，不生成可部署 binding。原生 IK 数值不稳定、无避碰保证等限制不通过适配修改。
- 标定支持 `virtual_wrist_sharpa_wave_bimanual`、`virtual_wrist_sharpa_wave_right`。支持 task v1/v2/v3 的具名单／多 free root、被动 hinge 与固定环境。关节与 actuator 独立映射；仅接受 unit-gear、无内部动态的独立位置 actuator。缺失映射、观测或身份不匹配直接报错。
- 原生 sampling、reward 和噪声公式保持；按名字映射 wrist/finger/object 权重，被动 hinge 使用 object rotation 权重。全部动态物体参与跟踪。保留原生 NaN 奖励／采样缺陷。
- 通用入口关闭 contact guidance/reward，`num_dr=num_dyn=1`。MPC 保留共同模型质量、惯量、碰撞、弹簧、重力补偿与伺服；仅改变私有 timestep。容量保持 `nconmax=256`、`njmax=1024` 以容纳共同模型，不照搬小原生资产的 120／350。
- prepared 与 plan 封存来源、配置及时间映射；更新配方必须重新 prepare／plan。默认每序列一个规划 seed；sim／sim-dr 复用同一控制，评估扰动不触发离线重规划。

通信协议为 `h2r-evaluation-v1`：4-byte 大端长度 + JSON，`hello/start/frames/step`。仅返回 `joint_targets`，没有 root wrench、物体辅助力、评估 reset 或方法提前终止。每次 eval 的 binding 与控制日志记录相同 `plan_id`，规划耗时与 eval 耗时分别保存。

## 检查与已验证范围

```bash
python -m unittest discover -s tests -p 'test_h2r*.py' -v
```

回放测试覆盖 20 个独立 episode 的相同控制、扰动观测不影响输出、末尾不足一个控制周期、游标检查、不完整／非有限计划及文件篡改拒绝。

边界测试覆盖乱序 actuator、多 root+hinge、body 原点速度、源姿态转换、端点、封存完整性和多 episode 终止。兼容单手布局下，原生与适配 reward/terminal reward、固定随机输入的控制采样逐值一致。新增回归覆盖私有 IK 约束、过滤有效时间、完整端点、多动作执行与移窗、下一步 lookahead、在线观测时机和非整比发布。

历史在线 MPC 运行结果保留在 bench 的 `runs/checks/spider-h2r-20261107/` 及对应 evaluation batch 中，不代表离线控制回放结果。各数据集离线规划与物理评估的实际通过范围，以新批次运行记录为准。

## 显式在线诊断

`evaluation --prepared PREPARED --device cuda:0` 仍可显式运行在线 MPC，每个 eval episode 在 0.4 秒边界根据真实观测重新规划，期间发布已有控制。重规划边界必须精确落在共同发布网格上，否则拒绝；不会静默取整。`spider.h2r.check` 只适用于这种在线名义记录。此模式与默认的一次规划／重复回放分开记录，不能混报成绩。

## 任务与模型边界检查

测试覆盖 task 资源篡改、重新封存后的任务身份变化、prepared 添加模型／路径或伪造 hash、
包外依赖与 XML 隐式资源目录拒绝。数值检查对照原 task 编译模型，确认 tracking sites 与规划 timestep
不会改变质量惯量、关节、碰撞、mesh、执行器或共同 solver 参数。

历史 Leap wrench、JSON-lines 协议、原生数据集／场景生成及 `spider-mjwp` 入口已移除。
原生复现使用上游 checkout；已有运行记录不在本次代码迁移中改写。
