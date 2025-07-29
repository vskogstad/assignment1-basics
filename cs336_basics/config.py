from dataclasses import dataclass

@dataclass
class Config:
    
    def __init__(self, path: str | None=None):
        """reads config from file or initializes with the variables shown below"""
        
        if path:
            raise NotImplementedError()
        else:
            optimizer = 

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer", type=torch.optim.Optimizer, help="Optimizer to use"
    )


@dataclass
class Config:
    name: str = "transformer"
    vocab_size: int = 10000
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 6
    d_ff: int = 128
    dropout: float = 0.1
    max_seq_len: int = 1024
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
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
    use_wandb: bool = True
    wandb_project: str = "lm_training"


class Config:
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, glu=False):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.context_length = context_length
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.glu = glu