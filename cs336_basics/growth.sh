#!/bin/bash
uv run cs336_basics/train_model.py --config cs336_basics/configs/stack.yaml --experiment_name growth_exp6
uv run cs336_basics/checkpoints.py
uv run cs336_basics/train_model.py --config cs336_basics/configs/stack_large.yaml --from_checkpoint grown.pth --wandb_resume growth_exp6
