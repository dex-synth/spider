# SPIDER Agent 指南

本 fork 是 h2r_bench 专用 baseline。使用入口与边界见 [README](README.md) 和 [H2R 说明](docs/h2r.md)。

- 机器人、物体、环境与示范只从 bench 的完整 task 读取；不要恢复自带资产、原生场景生成或模型路径 fallback。
- task 只经 `spider/h2r/h2r_io/`（`h2r-bench export-loader` 的副本）读取；不修改该目录，缺少的能力在方法代码中处理并反馈给 bench。向 `Layout`、`scene`、`reference` 等传递 `h2r_io.Task` 对象，不重建旧式 task 字典。
- IK、MPC、标定、奖励与控制算法留在 SPIDER；bench 独立推进评估物理并计分。
- `spider.h2r.scene` 仅在内存中添加无质量 tracking sites；MJWarp 只改变明确配置的私有积分 timestep。默认使用 `preprocess/ik_fast.py` 的 Mink 流水线；私有初态稳定性检查使用上游 .002 s timestep，不改变共同评估模型。
- prepared 保存方法参考，不保存物理场景。产物格式为 `spider-h2r-v5`／`spider-h2r-plan-v5`，绑定 `task_id`、`demo_id`、`robot_id`；更早的格式明确拒绝，不能静默迁移。
- 使用独立 Python 3.12 环境与本仓库依赖。开发环境见 `scripts/h2r_environment.sh`，不要升级共享环境中的其他方法依赖。
- 修改输入／模型边界时验证 task 文件闭包、物理参数保持和拒绝路径；修改算法时检查原生语义。
- Baseline 只修正相对所选上游流水线的适配错误；原生已有的算法缺陷、无碰撞避免约束等保持原样，不以提升成绩为由增加算法能力。原论文入口与后来的 fast 变体须分别标明，不能混合其步骤或默认值后声称完整复现。
- 当前默认为上游 fast：按数据集读取固定 YAML，最后一轮取最佳样本并从快照推进，保留重试／回退。明确授权的例外是 `num_dr=1` 不施加上游非零参数偏移，以及 free-root 四元数的符号对齐、平滑后归一化与 NLERP；保持窗口、插值位置、时钟及独立速度插值，不顺带修其他原生缺陷。
- prepared 须带 `pipeline.rotation_processing=sign-continuous-mean-nlerp-v1`，缺失或未知时重新 prepare。
- 时钟各归其主：plan 以方法的原生动作周期（`control_dt_s`，0.01 s）发布，不保存物理 tick；评估物理步长只取自 `hello.dt_s` 或 rollout 的 `run.json`，不取自 task。初始穿透预检使用方法侧常数，不从 task 读取。
- 数据集配方在 `settings.DATASET_RECIPES` 中逐一列出（arctic、oakink2、hot3d、taco），未知数据集报错；按 `task.hand_sides` 处理单手与双手，不按关节数或机器人变体名判断。
- 聚焦检查：`python -m unittest discover -s tests -p 'test_h2r*.py' -v`。
- 代码使用 Ruff、88 列、Google docstrings；面向 Agent 的说明使用中文。
