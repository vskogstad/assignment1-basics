from dataclasses import dataclass


@dataclass
class Config:
    name: str = "transformer"
    vocab_size: int = 10000
    num_layers: int = 6
    num_heads: int = 4
    d_model: int = 64
    d_ff: int = 128
    context_length: int = 1024
    theta: int | None = None

    # data loading
    train_dataset_path: str = "data/test_array2.npy"
    val_dataset_path: str = "data/validation.npy"

    # Training
    dtype: str = "Float32"
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    total_steps: int = 100000
    grad_clip_norm: float = 1.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"

    # Experiment
    name: str = "baseline"
    seed: int = 42
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = "outputs"
    #use_wandb: bool = True
    wandb_project: str = "lm_training"

