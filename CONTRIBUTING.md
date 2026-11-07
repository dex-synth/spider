# Contributing

This fork runs SPIDER on h2r_bench tasks. Read [README](README.md) and
[the H2R interface](docs/h2r.md) before changing task or model loading.
Native workflows are maintained in the upstream repository.

Use Python 3.12 and the independent environment:

```bash
uv sync --extra h2r --extra dev
uv run python -m unittest discover -s tests -p 'test_h2r*.py' -v
uv run ruff check spider/h2r tests spider/preprocess/ik_fast.py
uv run ruff format --check spider/h2r tests spider/preprocess/ik_fast.py
```

Keep algorithm changes explicit and preserve native sampling and IK semantics.
Changes to model loading must verify physical equivalence and reject altered or
external resources. A short independent bench rollout checks deployment;
planning traces and kinematic playback do not establish task success.

This project retains the upstream [CC BY-NC license](LICENSE).
