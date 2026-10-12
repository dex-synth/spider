# h2r_bench 接入

本入口是显式任务适配，算法继续由本仓库实现。原生 ARCTIC／XHand／底部刚体配方与共同 LeapHand／完整铰接剪刀的结果分别报告。

在专用容器的 SPIDER 环境执行，`H2R_REFERENCE` 指向 bench 的源参考包，`H2R_ASSET` 指向 bench 生成的共同资产，`H2R_PREPARED` 为新的输出目录：

```bash
/opt/h2r/spider/bin/python scripts/h2r_prepare_arctic.py \
  --bundle "$H2R_REFERENCE" --asset "$H2R_ASSET" --output "$H2R_PREPARED" \
  --collision-avoidance --max-ik-step-norm 0.005
/opt/h2r/spider/bin/python scripts/h2r_controller_server.py \
  --prepared "$H2R_PREPARED" --socket "$H2R_SOCKET" \
  --samples 128 --iterations 1 --seed 0 --device cuda:0
```

随后由独立 bench 进程调用 `h2r-bench evaluate`，初始化使用 `$H2R_PREPARED/initial.json`。控制器仅接受一次常驻连接；关闭后需重新启动。

已验证范围：共同 `s02/scissors_use_01 [350,451)`，右 LeapHand、1× 双部件剪刀、MuJoCo Warp 规划、独立 CPU MuJoCo 评估。开发运行完成全时域物理执行，但物体掉落、跟踪失败；不能称为论文性能复现或正式 benchmark 成绩。

## 配方差异

- 复用 `preprocess/ik_fast.py` 的 Mink wrist／finger／posture tasks；新增四指 Leap、浮动根部和 object hinge 的资产映射。
- 原生 fast 默认不处理碰撞。共同任务的无碰撞约束版本初始化穿透约 16.8 mm，被 bench 拒绝。
- 上述命令显式启用几何碰撞约束、冻结对象 IK 自由度、限制 IK 更新范数，属于算法变体。Mink 碰撞界采用位移单位，以匹配其优化变量 `delta_q`；不修改安装环境中的共享 Mink 代码。所有新开关默认关闭，原生默认行为保留。
- 共同入口保留首尾全部源帧，平均滤波窗口为 1；重采样到 120 Hz 时对 freejoint quaternion 使用 Slerp，不改变物理时间。
- 浮动根部采用 MuJoCo site position servo，输出该控制器在当前真实状态上的世界坐标 wrench；手指仍输出位置目标。位置／旋转增益分别为 300/30 与 3/0.3。私有模型有 actuator effort bounds，实际执行仍服从 bench 的范数限制。
- 原生 sampling optimizer 和目标跟踪 reward 被复用；新增布局包含完整 object hinge，使用旋转项相同的权重。没有物体辅助执行器。
- 128 样本、1 iteration、0.2 秒窗口仅为开发配置。每四个 120 Hz 控制步重规划；内部每个样本控制步包含四个 480 Hz 积分步。末尾仅在规划器内部 hold 最后目标并置零参考速度，不延长评估。

## 协议

`h2r-state-control-v2` 为 Unix socket JSON lines：`describe`、`reset`、`control`、`close`。状态包含 `robot`／`object` 的 joint names、joint positions／velocities，以及 13 维 `root_link_pose_com_velocity`：link 的 xyz、wxyz、根 body COM 的世界线速度、世界角速度。必填 `root_com_local_m` 给出该 COM 在 link 中的位置，转换可保留跨资产的 link twist 与 wrench 力矩；不能把 link 原点速度当作 COM 速度。

`control` 按连续 tick 和物理时间调用。响应只有手指目标和根部 wrench；每次调用均根据实际状态解码伺服，即使该步没有重规划。私有规划状态可以同步／搜索，方法无权推进、reset 或改写 bench 世界。

聚焦检查：`/opt/h2r/spider/bin/python -m unittest discover -s tests -p test_h2r.py -v`。

部署核对使用 `scripts/h2r_controller_check.py --prepared ... --trace .../rollout.jsonl --output ...`：在真实八步状态序列中确认规划初态同步、COM 速度往返、原生执行器到共同指令的转换和 tick 0/4 重规划。根部广义力误差约 `7.1e-15`，手指 effort 误差约 `3.3e-16`。固定控制计划后，非重规划 tick 的根部位置扰动 1 mm 使反馈力改变 0.3 N。

Reset 会清空私有 Warp 状态，并重置随机种子及初始控制计划。状态／随机状态／初始计划能复原，但原生 GPU 优化重复运行仍观察到输出差异；当前不承诺逐位确定性。差异单独记录，不能把数值转换一致性解释为 MPC 搜索或最终任务成绩完全可复现。
