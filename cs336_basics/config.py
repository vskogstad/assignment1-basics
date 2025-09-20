import argparse
import os
import torch
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
    pre_norm: bool
    glu: bool
    layer_norm: bool
    num_kv_groups: int | None

    # data loading
    train_dataset_path: str
    val_dataset_path: str
    tokenizer_vocab_path: str
    tokenizer_merges_path: str

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
    muon_momentum: float
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
        config_dict = asdict(self)
        
        # Handle torch dtype objects
        if 'dtype' in config_dict:
            dtype_value = config_dict['dtype']
            if hasattr(dtype_value, '__module__') and 'torch' in str(dtype_value.__module__):
                # Convert torch dtype back to string
                config_dict['dtype'] = str(dtype_value).split('.')[-1]  # torch.bfloat16 -> bfloat16
            elif hasattr(dtype_value, '__name__'):
                config_dict['dtype'] = dtype_value.__name__
        
        with open(file=filepath, mode="w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

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
    parser.add_argument("--max_learning_rate", type=float)
    parser.add_argument("--min_learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--eps", type=int)
    parser.add_argument("--betas", type=tuple)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--pre_norm", type=bool)
    parser.add_argument("--layer_norm", type=bool)
    parser.add_argument("--glu", type=bool)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--context_length", type=int)
    

    return parser