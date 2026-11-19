# SPIDER for h2r_bench

本 fork 用于 [h2r_bench](https://github.com/dex-synth/h2r_bench) 的共同任务评估。任务、机器人、物体、支撑与评估物理由 bench 提供。
有两个配方：正式配方是上游 CPU workflow（`dev/cpu`，MuJoCo CPU 批量接口上的 sampling MPC，见 [CPU 配方](docs/h2r-cpu.md)）；可选配方对齐上游 fast Mink IK 与 best-sample MPC，下文“使用”一节描述的是它。
原生数据集处理、场景生成、自带机器人资产和旧 Leap 部署入口已移除；原生配方请使用
[上游 SPIDER](https://github.com/facebookresearch/spider)。

```text
CPU 配方：  bench task → V2D Sharpa IK（由 FlashCHORD 的 prepare 导入）→ 私有 MuJoCo CPU MPC → 封存机器人控制
fast MJWarp：bench task → SPIDER fast Mink IK 与原有平滑 → 私有 MJWarp MPC → 封存机器人控制
    两者的控制都交给 bench 独立物理评估
```

## 使用

Python 3.12，独立环境：

```bash
uv sync --extra h2r
python -m spider.h2r.prepare --task "$TASK" --output "$PREPARED"
python -m spider.h2r.plan --task "$TASK" --prepared "$PREPARED" --output "$PLAN" --seed 0 --device cuda:0
python -m spider.h2r.evaluation --task "$TASK" --plan "$PLAN" --socket "$SOCKET" --output "$CONTROL_LOG"
python -m spider.h2r.check --task "$TASK" --plan "$PLAN" --run "$ROLLOUT" --output "$REPLAY_CHECK"
```

`TASK` 是 bench 分发的 task 目录（`h2r-task-v1`，独立打包或位于 suite 内），也是唯一的输入；
由依赖包 `h2r-io`（bench 发布的加载器）读取并校验。prepare 和 plan 均从该任务构建内存中的模型，
IK 与 MPC 模型只增加无质量 tracking sites；不增加 IK 碰撞避免或速度约束。
prepared 不携带 XML、mesh 或可覆盖场景的路径。
每个序列规划一次，sim 与 sim-dr 的评估 seeds 复用同一 plan。
手部标定按机器人名称选择（当前为 `sharpa_wave`），只处理 task 的参与手；其他机器人明确拒绝。
采用固定上游 fast 入口：ARCTIC、OakInk2 使用各自的 fast 配方，HOT3D、TACO 没有官方 override，使用基础默认值；其他数据集报错。
私有规划与发布的动作周期均为 0.01 s，重规划间隔按数据集配方；客户端在 start 中声明该周期，
评估物理步长只从 `hello.dt_s` 读取。
最佳样本快照、重试与回退仅作用于私有规划，不改变评估世界。

完整环境、评估命令、输入校验与支持限制见 [H2R 使用说明](docs/h2r.md)。
准备与计划格式为 `spider-h2r-v6`、`spider-h2r-plan-v6`；更早的格式需要从 bench task 重新准备与规划。

## 开发

- `spider/h2r/`：标定、参考、具名控制、方法产物与评估客户端。
- `spider/h2r/check.py`：CPU 检查 plan 与统一评估的实际提交控制、初态和时间覆盖，保留提前失败状态。
- `spider/h2r/cpu_scene.py`、`cpu_world.py`、`cpu_plan.py`：CPU 配方的私有模型与表、simulator 模块和规划入口；循环与判定是上游的 `examples/run_cpu.py`、`spider/simulators/cpu.py`、`spider/postprocess/evaluate_cpu.py`。
- `spider/preprocess/ik_fast.py`：Mink 分阶段求解及原生后处理。
- `spider/h2r/kinematics.py`：共同目标与 fast IK 的具名／时间映射。
- `spider/optimizers/sampling_fast.py`：原生最佳样本优化器；复用 `sampling.py`。
- `spider/h2r/fast_loop.py`：原生重试、boost、锚点回退与失败退出。
- `spider/simulators/mjwp.py`：私有 MJWarp 物理执行；接收 task 编译模型，不接受模型路径。
- `spider/config.py`：优化器数值配置；公开配方由 `spider/h2r/settings.py` 限定。

```bash
uv run --extra h2r python -m unittest discover -s tests -p 'test_h2r*.py' -v
```

## License

SPIDER is released under the Creative Commons Attribution-NonCommercial 4.0 license. See `LICENSE` for details.

## Code of Conduct

We expect everyone to follow the Contributor Covenant Code of Conduct in `CODE_OF_CONDUCT.md` when participating in this project.

## Upstream acknowledgments

- Thanks Mandi Zhao for the help with the [DexMachina workflow](https://github.com/MandiZhao/dexmachina) for SPIDER + Genesis.
- Thanks Taylor Howell for the help in the early stages of integrating [Mujoco Wrap](https://github.com/google-deepmind/mujoco_warp) for SPIDER + MJWP.
- Thanks Haoyang Weng for the help with the [HDMI workflow](https://github.com/lecar-lab/hdmi) for SPIDER + Sim2real RL.
- Inverse kinematics design is ported from [GMR](https://github.com/YanjieZe/GMR) and [LocoMujoco](https://github.com/robfiras/loco-mujoco).
- Dataset processing is ported from [Hot3D](https://github.com/facebookresearch/hot3d), [Oakinkv2](https://github.com/oakink/OakInk2), [Maniptrans](https://github.com/ManipTrans/ManipTrans), [Gigahands](https://github.com/Gigahands/Gigahands).
- Visualization inspired by other good sampling repo like [Hydrax](https://github.com/vincekurtz/hydrax) and [Judo](https://github.com/bdaiinstitute/judo).


## Citation

```bibtex
@article{pan2025spiderscalablephysicsinformeddexterous,
      title={SPIDER: Scalable Physics-Informed Dexterous Retargeting},
      author={Chaoyi Pan and Changhao Wang and Haozhi Qi and Zixi Liu and Homanga Bharadhwaj and Akash Sharma and Tingfan Wu and Guanya Shi and Jitendra Malik and Francois Hogan},
      year={2025},
      eprint={2511.09484},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2511.09484},
}
```
