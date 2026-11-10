# 相对 original SPIDER 的 H2R 适配

本文记录当前 H2R v4 实现相对 original 的差异，核对日期为 2026-11-10。

| 对照对象 | 版本与范围 |
| --- | --- |
| original | `facebookresearch/spider@44717007de41cbef7565dff7ff9f4453557a2d3d`；本机 `baselines/spider` |
| H2R 兼容版 | `dex-synth/spider`，v4 + `sign-continuous-mean-nlerp-v1`；本机 `h2r_bench_baselines/spider`，具体 revision 封存在 binding 中 |
| 共同评估 | `h2r_bench` 封存的 task 与独立 physical rollout；新规范参考声明 `wxyz-unit-continuous-v1` |

选定的上游流水线是 `spider/preprocess/ik_fast.py` → `examples/run_mjwp_fast.py`。
H2R 保留其 fast IK、sampling MPC、最佳样本选择及重试／回退语义，部署为 **离线控制回放**（`offline-control-replay`）。
这不是普通 `run_mjwp.py`／论文入口的复现，也不是消费真实评估观测的在线 MPC。运行命令见 [H2R 使用说明](h2r.md)。

## 实现与接口改动

| 项目 | original | H2R 兼容版 |
| --- | --- | --- |
| 任务输入 | 方法读取原始数据集、MANO 资产和处理目录 | `prepare --task` 读取 bench 封存的示范、任务和机器人 bundle；直接使用已解码 MANO joints 与 root rotation |
| 场景资产 | 方法生成 XML、使用自带机器人与物体资产 | IK、MPC 从同一 task 编译私有模型，只添加无质量 tracking sites；无自带场景 fallback，保留 context 物体 |
| 机器人映射 | 依赖 embodiment 和固定状态切片 | 按 task 的关节、actuator、free root、被动 hinge 名称映射；reward 公式不变，被动 hinge 使用物体旋转权重 |
| 手部标定 | 原生机器人 tracking sites | 新增 Sharpa 双手／右手虚拟腕的 MANO-to-wrist 固定旋转；参数在 `spider/h2r/config/sharpa_wave.json`，其他机器人明确拒绝 |
| 配置入口 | Hydra／原生数据集配置与模型路径 | `spider/h2r/settings.py` 只接收 `ik`、`mpc` 算法字段；任务、机器人、资产与速度由 bench 指定 |
| 执行入口与输出 | `run_mjwp_fast.py` 运行原生 fast 流程并保存轨迹 | `spider.h2r.plan` 封装相同 fast 流程，输出绑定共同 task、可供独立评估回放的封存 plan |
| 评估部署 | 原生 rollout 与方法内计分 | `evaluation` 通过 `h2r-evaluation-v1` 提交具名 `joint_targets`；bench 独立推进物理、施加扰动、终止并计分 |
| 产物校验 | 原生轨迹与日志 | 新增 task／plan 身份、文件 hash、控制时钟与完整覆盖校验；prepared 不保存场景，旧 v1–v3 产物必须重新生成 |
| 回放诊断 | 无对应统一评估检查器 | `spider.h2r.check` 在 CPU 核对 plan、初始化、提交控制及 physical JSONL／NPZ；区分有效失败前缀和完整执行 |

控制器不能提交物体状态、辅助力、评估 reset 或提前终止。规划 seed 与评估 seeds 分开；同一序列的 `sim`、`sim-dr` 复用同一 plan，回放控制不随评估观测改变。
私有规划的 `qpos/qvel` 只作诊断，实际成绩来自 bench 的 physical rollout。

## 保留的算法与默认参数

`spider/h2r/config/default.yaml` 和四份 `override/*_fast.yaml` 与上述 original **逐字节一致**。
解析顺序为基础默认值、对应数据集 fast override、任务参考采样周期、显式 `--config`；只提取算法字段，实际值写入 `resolved_config.yaml`。
HOT3D 没有专用 override，使用基础配方。以下均为未传自定义配置时的值。

### IK

| 参数／步骤 | original fast 与 H2R 共用值 |
| --- | --- |
| 求解器与积分 | Mink／DAQP；`sim_dt=0.005 s`，每帧 `max(1, round(ref_dt / sim_dt))` 步 |
| 初始收敛 | wrist-only 200 步，再加入 fingertips 300 步 |
| 目标权重 | wrist position `0.3`、wrist orientation `3.0`、finger position `10.0`、posture `0.01` |
| 阻尼 | FrameTask `lm_damping=1.0`；求解器 `damping=1e-5` |
| 参考处理时序 | 裁掉原始最后一帧；3 帧 valid 平均，再差分并丢弃首帧；H2R 的四元数数值修正见下文 |

物体 qpos 在 IK 每步直接写入方法参考。H2R 在 free-root 四元数平滑前归一化并对齐符号、平均后重新归一化，再差分 qvel；三处 qpos 插值保留原位置与分量线性方式，但在插值前对齐符号、插值后归一化，形成 NLERP。平移、scalar joints、ctrl 与独立 qvel 插值保持原处理；不新增 SLERP、碰撞避免或速度约束。
这项明确授权的数值修正避免 `q/-q` 抵消与非单位四元数，不能称为与 original 逐值等价。近零平均和非法源四元数明确拒绝，不填 identity。它仅作用于方法私有参考，不改写共同物体初态或目标。
带手指控制噪声 `0.2` 的 IK 前向 rollout 仅作诊断，不作为 MPC 参考。

### MPC

上游 fast 与 H2R 均选择最佳样本，复用其控制、post-step 状态和快照推进，并保留重试／回退；均不重新仿真均值控制。这些属于保留的算法行为，接口差异是上表中的入口封装与产物交接。

所有配方均使用 `num_samples=1024`、`sim_dt=0.01 s`（100 Hz 动作）。`ctrl_dt` 是每次优化后执行的时长／重规划间隔，不是单个动作的保持时间。

| 数据集 | fast override | iterations | `ctrl_dt / horizon / knot_dt`（s） | temperature | 物体旋转门槛（rad） | sanity check（s） | contacts／constraint rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HOT3D／无专用配方 | 基础 default | 32 | `0.4 / 1.6 / 0.4` | 0.1 | 0.3 | 3 | 120／350 |
| ARCTIC | `arcticv2_fast` | 4 | `1.5 / 3 / 1` | 1 | 2.5 | 0 | 256／768 |
| OakInk2 | `oakinkv2_fast` | 4 | `1.5 / 3 / 1` | 1 | 1.0 | 3 | 120／350 |
| GigaHand | `gigahand_fast` | 8 | `1.5 / 3 / 0.5` | 1 | 0.5 | 3 | 120／350 |
| OakInk | `oakink_fast` | 8 | `1.5 / 3 / 1` | 1 | 0.5 | 3 | 120／350 |

噪声和 reward 同样沿用对应配方，不只对齐 samples／iterations：

| 参数 | 基础 default | ARCTIC／OakInk2／OakInk | GigaHand |
| --- | --- | --- | --- |
| `first_ctrl_noise_scale / last_ctrl_noise_scale / final_noise_scale` | `0.5 / 1 / 0.1` | `2 / 4 / 0.01` | `1 / 2 / 0.01` |
| `joint_noise_scale / pos_noise_scale / rot_noise_scale` | `0.15 / 0.03 / 0.03` | `0.1 / 0.01 / 0.01` | `0.03 / 0.03 / 0.01` |
| `base_pos_rew_scale / base_rot_rew_scale` | `1 / 0.3` | `0.1 / 0.03` | `0.1 / 0.03` |

各配方共同保留：`joint_rew_scale=0.003`、物体 `pos_rew_scale=1`／`rot_rew_scale=0.3`、`vel_rew_scale=0.0001`、`terminal_rew_scale=1`、`exploit_ratio=exploit_noise_scale=0.01`、物体位置门槛 `0.1 m`。
sanity check 在私有 CPU 模型以 `0.002 s` 积分，检查全过程最大 root 漂移，门槛 `0.01 m / 0.3 rad`；ARCTIC 按原生 override 关闭。

fast 每个 attempt 跑满配置轮数，最多 4 个 attempt，不使用 improvement 提前停止。runner 保留最多 3 次 local retry、恢复上限 `max_revert_depth=3`；每级 boost 将 iterations 乘 2、noise 乘 `sqrt(2)`，恢复时回到同一锚点。普通成功不清除 boost，越过恢复锚点后才重置。
失败保留截断／最后候选诊断，但不发布不完整 plan。`contact_guidance`、`gibbs_sampling`、`terminate_resample` 默认均为 `false`；前两项开启时明确拒绝，符合所选上游 fast 的限制。

## 明确不同的参数与时间边界

| 项目 | original fast | H2R 处理与原因 |
| --- | --- | --- |
| 标称规划 | `num_dr=num_dyn=1`；`linspace(-0.005, 0.005, 1)` 构造 `xy_offset=pair_margin=-0.005 m` | 数量仍为 1，但不施加这两个方法侧偏移，沿用 task 标称参数；这是显式保留的算法例外，其他 DR 数量拒绝。评估 `sim-dr` 由 bench 独立施加 |
| 私有物理步长 | `sim_dt=0.01 s` | 默认不变；新增 `physics_substeps=1`，私有积分步长为 `sim_dt / physics_substeps` |
| hand solver／接触覆盖 | 强设 `iterations=20`、`ls_iterations=50`、`o_solref=[0.02,1]`、`o_solimp=[0,0.95,0.03,0.5,2]`、`implicitfast` | 删除这些方法侧覆盖，继承 task 的 solver、接触、惯量、执行器与限幅；规划模型只改变私有 timestep |
| 初始状态 | 从平滑参考的 qpos/qvel 初始化，先执行一次 `mj_step` 再将时间归零 | 使用共同原始物体姿态、IK 机器人位置、零速度；实际起点无 warm-up。新增关节限位和 task 穿透门槛检查，失败不生成可部署 binding |
| `ref_dt` 来源 | `task_info.json`，缺省 `0.02 s` | 从共同示范时间轴取得真实采样周期；例如本次 HOT3D 为 `1/30 s`，不强制改成 50 Hz |
| 裁剪／插值时间 | filtered 参考重新从零编号；先按基础 `0.02 s` 求 `ref_steps`，后读取数据集周期 | 保留原生插值因子与非旋转分量值，旋转使用 NLERP；显式记录窗口中心的源时间，再采样到私有网格；前后端点保持，速度独立插值、不在查询时重新差分 |
| 规划终点 | 默认 `len(ref) - horizon_steps - ctrl_steps` | 规划到共同 `task.end_tick`；末端参考保持，发布完整任务所需控制，不缩短共同计分时域 |
| 独立评估频率 | 无 H2R 评估时钟 | 新任务 `method=spider` 使用 300 Hz 物理／100 Hz 控制，每个原生动作保持 3 tick；最后一个动作可不足 3 tick |

因此，当前结果不能称为与原生场景、初态和错误计时逐帧等价。保留的是选定 fast 算法与配方，评估条件服从共同任务。
历史 240 Hz 物理／30 Hz 控制任务仍按其封存条件解释；零阶保持会丢弃部分 100 Hz 动作变化，新任务必须重新 prepare／plan。
plan 中的 `qpos/qvel` 对应控制执行后的 `source_time_s + native_action_dt_s`，不作为发布时刻的评估状态。
新规划要求 prepared 标记 `pipeline.rotation_processing=sign-continuous-mean-nlerp-v1`，缺失或未知时重新 prepare。文件结构仍为 v4；历史完整 v4 plan 保留原身份用于 CPU 回放与审计，不自动升级其旋转处理声明。

## 工程范围与验证边界

新增 `spider/h2r/` 承担任务读取、标定、具名映射、规划封存、协议客户端和回放检查。
`ik_fast.py` 抽出可调用的求解／后处理函数；`sampling_fast.py` 只新增可注入的 tracking-error 检查；`mjwp.py` 改为接收 task 编译模型；`fast_loop.py` 保留原 runner 的重试／回退。
原始数据处理、自带资产、场景生成、其他 simulator 工作流、viewer、旧计分／部署脚本及 `spider-mjwp` 入口已移除；原生运行使用 original checkout。

方法继续使用独立 Python 3.12 环境，不依赖 bench Python 包。新增 `h2r` extra 显式声明 SciPy、PyYAML、DAQP；本机 CUDA 12 安装配方固定 MuJoCo `3.7.0`、mujoco-warp `3.7.0.1`、Warp `1.12.1`、Torch `2.7.1`、Mink `1.1.0`，完整约束见 `scripts/h2r_runtime_constraints.txt`。评估环境独立，不要求与规划环境同版本。

旋转修正前的 v4 已通过 36 项测试。该历史版本的完整 HOT3D keyboard／Sharpa 双手任务使用默认规划配方，在 `sim`、`sim-dr` 各验证 seed 0，同一 plan 的 504 个控制均完整执行并通过共同评估；证据位于 bench 的 `runs/checks/spider-v4-replay-20261109/`。该结果不代表新旋转处理或所有数据集、机器人及多 seed 成绩。

NLERP 修正通过 40 项 H2R 测试及 ARCTIC `s01/box_use_01` 的 CPU 数组回归：问题窗口的完美源姿态跟随误差从 `3.138874 rad` 降至 `0.012080 rad`，原 tracking check 不再错误拒绝；非旋转位置、控制和对应差分速度保持一致。记录位于 bench 的 `runs/checks/spider-quaternion-nlerp-20261110/`。尚未重新运行该修正后的完整 plan／evaluation。
