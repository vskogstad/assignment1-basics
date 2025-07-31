import argparse
import os
from dataclasses import asdict, dataclass

import yaml


@dataclass
class Config:
    model_name: str
    vocab_size: int
    num_layers: int
    num_heads: int
    d_model: int
    d_ff: int
    context_length: int
    theta: int | None

    # data loading
    train_dataset_path: str
    val_dataset_path: str

    # Training
    dtype: str
    batch_size: int
    min_learning_rate: float
    max_learning_rate: float
    weight_decay: float
    warmup_steps: int
    total_steps: int
    cosine_cycle_steps: int | None
    betas: tuple
    eps: float
    grad_clip_norm: float
    optimizer: str
    scheduler: str

    # Experiment
    experiment_name: str
    seed: int
    log_interval: int
    eval_interval: int
    save_interval: int
    output_dir: str
    wandb_project: None | str
    from_checkpoint: None | str

    @classmethod
    def from_yaml(cls, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found at {filepath}")

        with open(file=filepath) as f:
            data = yaml.safe_load(f)
            

        return cls(**data)

    def save(self, filepath: str):
        with open(file=filepath, mode="w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def update_from_args(self, args):
        for command, value in vars(args).items():
            #print(command,value)
            if command == 'config':
                continue
            assert hasattr(self, command)
            if value is not None:
                setattr(self, command, value)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file (required)")
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--from_checkpoint", type=str)
    parser.add_argument("--total_steps", type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--max_learning_rate", type=int)
    parser.add_argument("--weight_decay", type=int)
    parser.add_argument("--eps", type=int)
    parser.add_argument("--betas", type=tuple)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--wandb_project", type=str)

    return parser