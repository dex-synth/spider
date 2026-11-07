# h2r_bench bundle 与统一评估

`spider.h2r.prepare` 直接读取 h2r_bench preprocess 的物理 bundle、robot bundle 和固定 task，生成 SPIDER 的 IK 参考。`spider.h2r.evaluation` 是常驻客户端，每次收到真实评估观测后，用原生 MJWarp sampling MPC 重规划，返回完整机器人具名位置目标。统一 eval 默认使用 MuJoCo。

算法、标定与 MPC 都在本仓库；bench 持有评估世界、初始化、时钟、终止和共同指标。此入口属于显式 H2R 任务适配，不能当作原生 ARCTIC/XHand 配方或论文性能复现。

## 环境

使用独立 Python 3.12 环境，不需要安装 h2r_bench。常规安装使用 `uv sync --extra h2r`；本机 CUDA 12 配方使用已有约束：

```bash
bash scripts/h2r_environment.sh /path/to/spider-env
```

脚本安装 `.[h2r]`（显式声明 SciPy、PyYAML、DAQP），保持 MuJoCo 3.7.0、mujoco-warp 3.7.0.1、Warp 1.12.1、Torch 2.7.1、Mink 1.1.0。完整约束见 `scripts/h2r_runtime_constraints.txt`；它是当前 CUDA 12 配方，与原生 `uv.lock` 的默认平台解析分开。已有容器可直接使用 `/opt/h2r/spider/bin/python`，无需重装。

此入口使用 bundle 中已解码的 MANO joints 与原始 root rotation，不读取原始数据集或 MANO 模型。运行旧 raw-data processor 时才设置 `H2R_INSTALL_RAW_DATA=1` 安装 manotorch/Chumpy。

## 准备与部署

以下路径均由调用者提供；输入目录已封存，输出目录必须不存在。先在 bench 环境固定任务与指标：

```bash
h2r-bench prepare-evaluation --scene-bundle "$HUMAN" --robot-bundle "$ROBOT" \
  --physics-hz 240 --control-hz 30 --output "$TASK"
h2r-bench prepare-metrics task="$TASK" output="$SPEC"
```

然后在 SPIDER 环境准备：

```bash
python -m spider.h2r.prepare --human-bundle "$HUMAN" --robot-bundle "$ROBOT" \
  --task "$TASK" --output "$PREPARED"
```

可选 `--config recipe.yaml` 覆盖 `ik` 或 `mpc` 的字段；字段及默认值见 `spider/h2r/settings.py`，未知字段会报错。配置改变需要新的 prepared 目录。初始化碰撞检查失败时保留诊断，但不生成可部署的 `h2r_binding.json`。

| 产物 | 用途 |
| --- | --- |
| `trajectory_keypoints.npz` | 同一时间轴上的 wrist 与 fingertip targets |
| `trajectory_kinematic.npz` | 全部源帧的 `time_s/qpos/qvel/ctrl`；ctrl 按 actuator 顺序 |
| `planning/` | 共同任务场景与资源，仅增加 tracking sites |
| `initial_robot.json`、`initialization.json` | 具名零速度初态提案与穿透检查 |
| `resolved_config.yaml`、`calibration.json` | 实际配方与 Sharpa wrist 标定 |
| `h2r_binding.json` | task/bundle/robot 身份、方法版本及产物 hash |

保留完整目录，可整体搬移；runtime 不依赖原始 human/robot 路径，但仍需匹配的 task。封存后不能直接修改其中的文件。

在 bench 的评估环境启动服务：

```bash
h2r-bench evaluate-server task="$TASK" spec="$SPEC" mode=sim backend=mujoco \
  num_trajectories=1 'seeds=[42]' socket="$SOCKET" output="$ROLLOUT" \
  timeout_s=900 video.enabled=false
```

待日志出现 `Evaluation ready:` 后，在 SPIDER 环境连接：

```bash
CUDA_VISIBLE_DEVICES="$GPU" python -m spider.h2r.evaluation \
  --prepared "$PREPARED" --task "$TASK" --socket "$SOCKET" \
  --device cuda:0 --output "$CONTROLLER_OUTPUT" --timeout-s 900
```

示例是单 seed 功能检查，关闭视频以独立验收控制链。标准批次固定预定 seeds，并分别运行 `mode=sim` 与 `mode=sim-dr`，使用独立的 socket 和输出目录。共同计分在 bench 环境执行：

```bash
h2r-bench score-evaluation --run "$ROLLOUT" --spec "$SPEC" --output "$SCORES"
h2r-bench summarize-evaluations --batch "$BATCH"
```

批次目录、预定集合与完整命令见 h2r_bench 的 `docs/methods.md`。完成时间轴不等于操作成功，最终成绩读取共同指标。

## 适配配方与边界

- 复用 `preprocess/ik_fast.py` 的 wrist/fingertip/posture tasks 和 DAQP 求解。冻结全部非机器人 DoF，保留全部帧与物体。当前配方不启用 collision-constrained IK；初态仍接受共同碰撞检查，不能通过移动物体或放宽阈值部署。
- 首批标定为 `virtual_wrist_sharpa_wave_bimanual` 与 `virtual_wrist_sharpa_wave_right`。腕部六个标量关节经真实运动学 IK 求解；机器人关节顺序与 actuator 顺序分别映射。其他机器人在 prepare 阶段明确拒绝。
- 支持物理 bundle/task v1/v2/v3 的具名单/多 free root、被动 hinge 与固定环境；仅接受 unit-gear、无执行器内部动态的独立位置 actuator。遗漏动态关节、额外 actuator、缺失观测、身份或时钟不匹配会报错。
- 默认开发配方为 128 样本、2 iterations、0.4 s horizon、0.1 s knot、8 个物理子步。动作周期与 task 的共同控制周期相同，每个目标在全部子步保持，每次控制观测都重规划；horizon/knot 必须落在共同控制网格上。
- 原生 sampling optimizer、加权 qpos 范数、qvel 项及 terminal multiplier 保持不变；按名字配置 wrist/finger/object 权重与噪声，被动 hinge 使用 object rotation 权重。所有动态物体都参与私有跟踪，包括 context 物体。
- 关闭 contact guidance/reward；`num_dr=1`，规划使用 nominal 参数。保留共同模型的质量、几何、弹簧、重力补偿与伺服；只按配置改变私有积分 timestep，不继承原生 hand 模式的 solver/contact 重写。评估 `sim-dr` 的真实扰动留在 bench，方法收到规定的带噪/延迟观测。
- 以真实 tick 采样未来参考，freejoint quaternion 用 SLERP，速度在相同网格上做流形差分。末尾 hold/零速度仅供私有 horizon，bench 的终点不变。
- 每个 episode 独立维护计划与 Torch RNG；终止 episode 不再发动作。每次搜索前清理私有 Warp 状态，写入真实 robot/object 状态、body 原点世界 twist 和有效目标，不使用旧 COM 修正或参考物体状态。

在线协议是 `h2r-evaluation-v1`：4-byte 大端长度 + JSON，`hello/start/frames/step`。仅返回 `joint_targets`，没有 root wrench、物体辅助力、评估 reset 或方法提前终止。Binding 记录准备耗时，控制日志分别记录首次 setup（含编译）与逐步计算耗时；此开发配置不承诺实时 30 Hz 或 GPU 逐位确定性。

## 检查与已验证范围

```bash
python -m unittest discover -s tests -p 'test_h2r*.py' -v
python -m spider.h2r.check --prepared "$PREPARED" --task "$TASK" \
  --run "$ROLLOUT" --controller-log "$CONTROLLER_OUTPUT/controller.jsonl" \
  --episode seed-42 --device cuda:0 --steps 3 --output "$CHECK_JSON"
```

检查工具对照名义 `sim` 的实际记录与保存的规划初态，再用连续状态重跑少量 native MPC，检查 GPU 世界同步、计划滚动、外部 RNG 保持与重复 tick 拒绝。它不写回评估或成绩；`sim-dr` 的真实状态不能代替带噪控制观测，故不接受其记录。

边界测试覆盖乱序 actuator、多 root+hinge、body 原点速度、源姿态转换、端点、封存完整性和多 episode 终止。兼容单手布局下，原生与适配 reward/terminal reward、固定随机输入的控制采样逐值一致。

本次实测包括 ARCTIC `s01/box_use_01` 的完整 889 帧双手准备，以及 MuJoCo `sim`/`sim-dr` 全时域闭环和共同计分；HOT3D `P0001_b2bcbe28_seg015__continuous` 右手 220 帧与 OakInk2 `scene_03/A004/seq__e177…` 双手 420 帧通过准备及初态检查，后两者尚未做物理评估。TACO `20231002_063` 完成双物体 IK，但初态手桌穿透超过共同 20 mm 阈值，准备明确失败，不能声称可部署。详尽运行记录由 bench 的 `runs/checks/spider-h2r-20261107/` 保存，不能推广为全数据集支持或性能声明。

## 历史入口

旧 `scripts/h2r_prepare_arctic.py`、`h2r_controller_server.py` 与 Leap wrench/JSON-lines 协议只用于历史产物，见 [h2r-legacy.md](h2r-legacy.md)。它们不兼容当前统一 eval；旧 viewer/check 脚本也不用于上述新 prepared 格式。
