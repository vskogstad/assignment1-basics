import argparse
import math
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict
from typing import IO, BinaryIO, Optional

# from einops import einsum, rearrange
import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import nn as nn

import wandb
from cs336_basics.config import Config, get_parser
from cs336_basics.model import Transformer
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.utils import resource_accounting

# TODO: Muon optimizer verified.
# Perplexity measurement.
# Train model on gpu
# Ablations and hyper-parameter search
#

def train(cfg: Config):
    """Main training loop
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer", type=torch.optim.Optimizer, help="Optimizer to use"
    )
    """
    # 
    if torch.cuda.is_available():
        max_flops = 989e12 / 2 # TF32 Tensor Core without sparsity.
    else:
        max_flops = float(inf)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cfg.dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
    
    # torch.set_default_dtype(dtype)
    print(f"Using {device} and {cfg.dtype}")
    print(f"Processing {cfg.batch_size * cfg.context_length * cfg.total_steps}")

    torch.set_float32_matmul_precision('high')
    torch.autograd.set_detect_anomaly(True)
    model = torch.compile(get_model(cfg, device))
    """for name, param in model.named_parameters():
        if 'rmsn' in name and 'weights' in name:
            param.requires_grad = False
            print(f"Frozen {name}")"""
    flops_per_batch = resource_accounting(cfg) # Print info about the current run to the log

    assert cfg.scheduler == "cosine"  # Not setup to use other schedulers yet.


    optimizer = get_optimizer(cfg=cfg, model=model)

    # Data loading
    train_data = np.load(cfg.train_dataset_path, mmap_mode="r")
    val_data = np.load(cfg.val_dataset_path, mmap_mode="r")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=cfg.tokenizer_vocab_path,
        merges_filepath=cfg.tokenizer_merges_path,
    )

    # Initialize logging
    if cfg.wandb_project:
        wandb.login()
        run = wandb.init(project=cfg.wandb_project, config=asdict(cfg))
    current_step = 0
    chunk_steps = 0
    chunk_loss = 0
    t_0 = time.time_ns()
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Load from checkpoint
    if cfg.from_checkpoint:
        source = os.path.join(cfg.output_dir, "checkpoints", cfg.from_checkpoint)
        step = (
            load_checkpoint(src=source, model=model, optimizer=optimizer) + 1
        )  # increment by one, this step has already been done

    # Training loop
    for step in range(current_step, cfg.total_steps):
        x, y = get_batch(
            dataset=train_data, batch_size=cfg.batch_size, context_length=cfg.context_length, device=device
        )

        optimizer.zero_grad()
        y_pred = model(x)
        #loss = cross_entropy(pred=y_pred, targets=y)


        # Trying to search for nan-source using regular cross entropy
        current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        print(f"step = {step}, LRs = {current_lrs}")
        print(f"y_pred min/max: {y_pred.min().item():.3f} / {y_pred.max().item():.3f}")
        print(f"y_pred contains inf: {torch.isinf(y_pred).any()}")
        print(f"y_pred contains nan: {torch.isnan(y_pred).any()}")
        loss = F.cross_entropy(rearrange(y_pred, "b s v -> (b s) v"), rearrange(y, "b s -> (b s)").long())

        loss.backward()

        clip_gradient(parameters=model.parameters(), max_l2_norm=cfg.grad_clip_norm)

        if cfg.scheduler == "cosine": # temporary disabled to test varying lr across groups
            new_lr = get_lr_cosine(
                step=step,
                max_learning_rate=cfg.max_learning_rate,
                min_learning_rate=cfg.min_learning_rate,
                warmup_steps=cfg.warmup_steps,
                cosine_cycle_steps=cfg.cosine_cycle_steps,
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        optimizer.step()

        # Logging
        chunk_loss += loss.item()
        chunk_steps += 1
        if step % cfg.log_interval == 0:
            loss_accumulated = chunk_loss / chunk_steps

            t_1 = time.time_ns()
            chunk_time = t_1 - t_0
            t_0 = time.time_ns()
            token_per_s = 1e6 * chunk_steps * cfg.batch_size * cfg.context_length / (chunk_time)
            print(
                f"step = {step} | Loss = {loss_accumulated:.3f} | ns {chunk_time = :.2f} | tok/ms = {token_per_s:.3f} | lr = {new_lr} | MFU = {(flops_per_batch / (chunk_time*1e-9)) / max_flops}"
            )

            chunk_loss = 0
            chunk_steps = 0
            if cfg.wandb_project:
                run.log({"Step": step, "Loss": loss_accumulated, "tok/ms": token_per_s, "lr": new_lr, "MFU": {(flops_per_batch / (chunk_time*1e-9)) / max_flops}})
        # Add this comprehensive monitoring every 25 steps
        if step % 25 == 0:
            print(f"\n=== Step {step} Weight Analysis ===")
            
            # 1. Embedding weights
            emb_std = model.embedding.embedding.std().item()
            emb_max = model.embedding.embedding.abs().max().item()
            print(f"Embedding: std={emb_std:.4f}, max={emb_max:.4f}")
            
            # 2. Attention weights (first few layers)
            for i in range(min(3, len(model.layers))):
                layer = model.layers[i]
                # Check attention projection weights
                for name, param in layer.mha.named_parameters():
                    if 'weight' in name or 'W' in name:
                        std = param.std().item()
                        max_val = param.abs().max().item()
                        print(f"Layer {i} MHA {name}: std={std:.4f}, max={max_val:.4f}")
                
                # Check FFN weights  
                for name, param in layer.ffn.named_parameters():
                    if 'weight' in name or 'W' in name:
                        std = param.std().item() 
                        max_val = param.abs().max().item()
                        print(f"Layer {i} FFN {name}: std={std:.4f}, max={max_val:.4f}")
            
            # 3. Output head
            head_std = model.lm_head.W.std().item()
            head_max = model.lm_head.W.abs().max().item()
            print(f"LM Head: std={head_std:.4f}, max={head_max:.4f}")
            
            # 4. Current output variance
            print(f"Y_pred range: [{y_pred.min().item():.3f}, {y_pred.max().item():.3f}]")
        # validation
        if step % cfg.eval_interval == 0 and step != 0:
            model.eval()
            with torch.no_grad():
                print(f"\nCalculating validation loss. Using batch size: {len(val_data)}")
                x_val, y_val = get_batch(
                    dataset=val_data, batch_size=cfg.batch_size, context_length=cfg.context_length, device=device
                ) # TODO Run eval on whole test-set.
                y_pred = model(x_val)
                val_loss = cross_entropy(pred=y_pred, targets=y_val)
                if cfg.wandb_project:
                    wandb.log({"Validation loss": val_loss})
                # sample from the model
                print(f"Sampling from the model at step {step} with validation loss {val_loss}")
                model.sample(tokenizer=tokenizer, prompt="It was a nice day")
            model.train()

        # checkpointing
        if step % cfg.save_interval == 0 and step != 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step,
                out=f"{cfg.output_dir}/checkpoints/{cfg.experiment_name}_{step}.pth",
            )

    if cfg.wandb_project:
        run.finish()


def get_optimizer(cfg: Config, model):
    """Returns an initialized optimizer with parameters from config"""
    # Currently only supports AdamW, but setup of parameters to also could initialize Muon optimizer later

    hidden_matrix_params = [
        param for name, param in model.layers.named_parameters() if param.ndim >= 2 and "embed" not in name
    ]
    embed_params = [param for name, param in model.named_parameters() if "embed" in name]
    scalar_params = [param for param in model.parameters() if param.ndim < 2]
    head_params = [model.lm_head.W]
    if cfg.optimizer == "adamw":
        optimizer = AdamW(
            [
                {"params": hidden_matrix_params, "weight_decay": cfg.weight_decay},
                {"params": embed_params, "weight_decay": 0},
                {"params": scalar_params, "weight_decay": 0},
                {"params": head_params, "weight_decay": cfg.weight_decay},
            ],
            lr=cfg.min_learning_rate,
            betas=cfg.betas,
            eps=cfg.eps,
        )
    elif cfg.optimizer == "muon":
        optimizer = MuonWithAdamW(
            [
                {"params": hidden_matrix_params, "weight_decay": cfg.weight_decay, "lr":0.05, "use_muon": True},
                {"params": embed_params, "weight_decay": 0, "lr":0.6, "use_muon": False},
                {"params": scalar_params, "weight_decay": 0, "lr":0.04, "use_muon": False},
                {"params": head_params, "weight_decay": cfg.weight_decay, "lr":0.22, "use_muon": False},
            ],
            #lr=cfg.max_learning_rate,
            momentum=cfg.muon_momentum,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            eps=cfg.eps,

        )
    else:
        raise NotImplementedError(f"No optimizer named {cfg.optimizer} has been implemented.")
    return optimizer


def get_model(cfg: Config, device):
    if cfg.model_name == "transformer":
        return Transformer(
            vocab_size=cfg.vocab_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
            context_length=cfg.context_length,
            theta=cfg.theta,
            device=device,
            dtype=cfg.dtype,
        )
    elif cfg.model_name == "silu-transformer":
        raise NotImplementedError()
    elif cfg.model_name == "pre-norm-transformer":
        raise NotImplementedError()
    elif cfg.model_name == "no-ln-transformer":
        raise NotImplementedError()
    else:
        raise NotImplementedError("Not implemented loading for model {cfg.model_name}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    # write to output path
    print(f"Saving checkpoint to {out}")
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    if not os.path.exists(src):
        raise FileNotFoundError(f"Checkpoint not found at {src}")
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Loading checkpoint from {src}")
    return checkpoint["iteration"]


def get_batch(dataset: np.array, batch_size: int, context_length: int, device: str):
    """Gets a random batch of x and y tensors for the model to train on"""

    if len(dataset) < context_length:
        raise ValueError("Dataset smaller than context length, not possible to create batches")

    # Generate random starting indices
    max_start_idx = len(dataset) - context_length - 1
    starts = np.random.randint(0, max_start_idx + 1, size=batch_size)

    # Create index arrays for vectorized sampling
    indices = starts[:, None] + np.arange(context_length + 1)

    batch = torch.from_numpy(dataset[indices]).to(device=device, dtype=torch.int32)  # .dtype(torch.int16)

    x = batch[:, :-1]
    y = batch[:, 1:]

    return x, y


def softmax(x: torch.Tensor, dimension: int):
    max_x = torch.max(x, dim=1, keepdim=True).values
    print(torch.max(x, dim=1))
    result = torch.exp(x - max_x) / torch.sum(torch.exp(x - max_x), dim=dimension, keepdim=True)

    return result


def cross_entropy(pred: torch.Tensor, targets: torch.Tensor):
    """Calculates the negative log likelihood"""
    pred = rearrange(pred, "b ... c -> (b ...) c")  # combine batch and eventula sequence dimension
    targets = rearrange(targets, "b ... -> (b ...)")
    batch_size = pred.size(0)
    max_pred = torch.max(pred, dim=1, keepdim=True).values

    pred_shifted = pred - max_pred  # Compute log-sum-exp: log(sum(exp(x_i)))
    log_sum_exp = torch.log(torch.sum(torch.exp(pred_shifted), dim=1, keepdim=True))

    # Get the logits for the target classes
    target_logits = pred_shifted[torch.arange(batch_size), targets].unsqueeze(1)
    loss_per_sample = log_sum_exp - target_logits

    return loss_per_sample.mean()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params={}, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or initial value.
                m = state.get(
                    "m", torch.zeros_like(p.data)
                )  # Get first vector momentum from the state, or initial value.
                v = state.get(
                    "v", torch.zeros_like(p.data)
                )  # Get second vector momentum from the state, or initial value.

                grad = p.grad.data  # Get the gradient of loss with respect to p.
                m = beta_1 * m + (1 - beta_1) * grad  # Update first moment estimate
                v = beta_2 * v + (1 - beta_2) * grad**2  # Update second moment estimate

                lr_t = lr * (math.sqrt(1 - beta_2**t) / (1 - beta_1**t))  # adjust lr
                p.data -= lr_t * (m / (torch.sqrt(v) + eps))  # Update weight tensor in-place.
                p.data -= lr * weight_decay * p.data  # Apply weight decay
                state["t"] = t + 1  # Increment iteration number.
                state["m"] = m
                state["v"] = v

        return loss


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=1e-4, eps=1e-7):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "eps": eps, "momentum": momentum, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            lr = group["lr"]  # Get the learning rate.
            momentum = group["momentum"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                # t = state.get("t", 1)  # Get iteration number from the state, or initial value.
                B_t = state.get("B_t", torch.zeros_like(p.data))

                grad = p.grad.data  # Get the gradient of loss with respect to p.
                B_t = momentum * B_t + grad
                O_t = newtonschulz5(B_t, steps=5, eps=eps)

                p.data -= lr * O_t  # Update weight tensor in-place.
                p.data -= lr * weight_decay * p.data  # Apply weight decay
                # state["t"] = t + 1  # Increment iteration number.
                state["momentum"] = B_t

        return loss

class MuonWithAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "eps": eps, "momentum": momentum, "weight_decay": weight_decay, "betas": betas}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            weight_decay = group["weight_decay"]
            if not group["use_muon"]:
                beta_1, beta_2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
            else:
                momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue
            
                if not group["use_muon"]: # Use adamw
                    state = self.state[p]  # Get state associated with p.
                    t = state.get("t", 1)  # Get iteration number from the state, or initial value.
                    m = state.get(
                        "m", torch.zeros_like(p.data)
                    )  # Get first vector momentum from the state, or initial value.
                    v = state.get(
                        "v", torch.zeros_like(p.data)
                    )  # Get second vector momentum from the state, or initial value.

                    grad = p.grad.data  # Get the gradient of loss with respect to p.
                    m = beta_1 * m + (1 - beta_1) * grad  # Update first moment estimate
                    v = beta_2 * v + (1 - beta_2) * grad**2  # Update second moment estimate

                    lr_t = lr * (math.sqrt(1 - beta_2**t) / (1 - beta_1**t))  # adjust lr
                    p.data -= lr_t * (m / (torch.sqrt(v) + eps))  # Update weight tensor in-place.
                    p.data -= lr * weight_decay * p.data  # Apply weight decay
                    state["t"] = t + 1  # Increment iteration number.
                    state["m"] = m
                    state["v"] = v

                else: # Use Muon
                    state = self.state[p]  # Get state associated with p.
                    B_t = state.get("B_t", torch.zeros_like(p.data))

                    grad = p.grad.data  # Get the gradient of loss with respect to p.
                    B_t = momentum * B_t + grad
                    O_t = newtonschulz5(B_t, steps=5, eps=1e-7)

                    p.data -= lr * O_t  # Update weight tensor in-place.
                    p.data -= lr * weight_decay * p.data  # Apply weight decay
                    # state["t"] = t + 1  # Increment iteration number.
                    state["momentum"] = B_t

        return loss


def newtonschulz5(G, steps=5, eps=1e-7):
    """From https://kellerjordan.github.io/posts/muon/"""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def get_lr_cosine(
    step: int, max_learning_rate: float, min_learning_rate: float, warmup_steps: int, cosine_cycle_steps: int
):
    """Returns a learning rate based on cosine annealing"""
    if step < warmup_steps:  # Warmup
        lr = step / warmup_steps * max_learning_rate
    elif step > cosine_cycle_steps:  # + warmup_iters: # Post Annealing
        lr = min_learning_rate
    else:  # Cosine annealing
        part = (step - warmup_steps) / (cosine_cycle_steps - warmup_steps) * math.pi
        lr = min_learning_rate + 0.5 * (1 + math.cos(part)) * (max_learning_rate - min_learning_rate)

    return lr


def clip_gradient(parameters, max_l2_norm: float):
    """Finds the size of the l2-norm, if higher than the max, we scale down."""
    eps = 1e-6

    # Calculate l2_norm
    norm_squared = 0
    for param in parameters:
        if param.grad != None:
            norm_squared += torch.linalg.vector_norm(param.grad) ** 2
    l2_norm = math.sqrt(norm_squared)

    if l2_norm <= max_l2_norm:
        return

    # Clip gradients
    print(f"clipping from {l2_norm}")
    for param in parameters:
        if param.grad != None:
            param.grad *= max_l2_norm / (l2_norm + eps)


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # load config
    config = Config.from_yaml(args.config)
    config.update_from_args(args)

    # Save final config to experiment directory
    os.makedirs(config.output_dir, exist_ok=True)
    config.save(os.path.join(config.output_dir, f"{config.experiment_name}_config.yaml"))

    # train the model
    train(cfg=config)

    # report results
