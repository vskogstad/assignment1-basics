from dataclasses import dataclass, asdict
import os
import yaml


@dataclass
class Config:
    name: str
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
    name: str
    seed: int
    log_interval: int
    eval_interval: int
    save_interval: int
    output_dir: str
    wandb_project: str

    @classmethod
    def from_yaml(cls, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found at {filepath}")
        
        with open(file=filepath) as f:
            data = yaml.safe_load(f)
            print(data)

        return cls(**data)

    
    def save(self, filepath: str):
        with open(file=filepath) as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def update_form_args(self, args):
        pass