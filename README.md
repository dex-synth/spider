# SPIDER for h2r_bench

本 fork 用于 [h2r_bench](https://github.com/dex-synth/h2r_bench) 的共同任务评估。
对齐 SPIDER 上游 fast Mink IK 与 best-sample MPC；任务、机器人、物体、支撑与评估物理由 bench 提供。
原生数据集处理、场景生成、自带机器人资产和旧 Leap 部署入口已移除；原生配方请使用
[上游 SPIDER](https://github.com/facebookresearch/spider)。

```text
bench task → SPIDER fast Mink IK 与原有平滑 → 私有 MJWarp MPC → 封存机器人控制
    └──────────────────────────────────────→ bench 独立物理评估
```

## 使用

Python 3.12，独立环境：

```bash
uv sync --extra h2r
python -m spider.h2r.prepare --task "$TASK" --output "$PREPARED"
python -m spider.h2r.plan --task "$TASK" --prepared "$PREPARED" --output "$PLAN" --seed 0 --device cuda:0
python -m spider.h2r.evaluation --task "$TASK" --plan "$PLAN" --socket "$SOCKET" --output "$CONTROL_LOG" --device cpu
```

`TASK` 必须是 bench 分发的完整、自包含任务。prepare 和 plan 均从该任务构建内存中的模型，
IK 与 MPC 模型只增加无质量 tracking sites；不增加 IK 碰撞避免或速度约束。
prepared 不携带 XML、mesh 或可覆盖场景的路径。
每个序列规划一次，sim 与 sim-dr 的评估 seeds 复用同一 plan。
当前标定为 Sharpa 双手和右手虚拟腕，其他机器人明确拒绝。
采用固定上游 fast 入口与各数据集的 fast 配方；HOT3D 没有官方 override，使用基础默认值。
默认私有规划 100 Hz，重规划间隔按数据集配方；共同控制发布采用零阶保持采样。
最佳样本快照、重试与回退仅作用于私有规划，不改变评估世界。

完整环境、评估命令、输入校验与支持限制见 [H2R 使用说明](docs/h2r.md)。
准备与计划格式为 `spider-h2r-v4`、`spider-h2r-plan-v4`；旧格式需要从原 bench task 重新准备与规划。

## 开发

- `spider/h2r/`：任务边界、标定、参考、具名控制与评估客户端。
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
