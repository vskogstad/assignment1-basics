import argparse
import math
import os
from collections.abc import Callable, Iterable
from typing import IO, BinaryIO, Optional

# from einops import einsum, rearrange
import numpy as np
import torch
from einops import einsum, rearrange
from torch import nn as nn

import wandb
from cs336_basics.config import Config
from cs336_basics.model import Transformer
from cs336_basics.tokenizer import Tokenizer

# TODO: training loop with config
# TODO: look for efficiency improvements. Mostly just passing tests at the moment
# TODO: Muon optimizer

'''
class Config:
    
    def __init__(self, path: str | None=None):
        """reads config from file or initializes with the variables shown below"""
        if path:
            raise NotImplementedError()
        else:
            optimizer = 


'''


def train(cfg: Config):
    """Main training loop
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer", type=torch.optim.Optimizer, help="Optimizer to use"
    )
    """
    # wandb.login()
    # run = wandb.init(project=cfg.wandb_project, config={})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(
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
    assert cfg.optimizer == "adamw"  # Not setup to use other optimizers yet.
    optimizer = AdamW(
        params=model.parameters(), lr=cfg.max_learning_rate, betas=cfg.betas, weight_decay=cfg.weight_decay, eps=cfg.eps
    )
    train_data = np.load(cfg.train_dataset_path, mmap_mode="r")
    val_data = np.load(cfg.val_dataset_path, mmap_mode="r")
    print(train_data[:15])

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Training loop
    for step in range(cfg.total_steps):
        print(step == 0)
        x, y = get_batch(
            dataset=train_data, batch_size=cfg.batch_size, context_length=cfg.context_length, device=device
        )
        optimizer.zero_grad()
        y_pred = model(x)
        loss = cross_entropy(pred=y_pred, targets=y)
        loss.backward()
        clip_gradient(parameters=model.parameters(), max_l2_norm=cfg.grad_clip_norm)
        optimizer.lr = get_lr_cosine(
            step=step,
            max_learning_rate=cfg.max_learning_rate,
            min_learning_rate=cfg.min_learning_rate,
            warmup_steps=cfg.warmup_steps,
            cosine_cycle_steps=cfg.cosine_cycle_steps,
        )
        optimizer.step()  # Pass in learning rate?

        # Logging and validation
        if step % cfg.eval_interval == 0:
            with torch.no_grad():
                x_val, y_val = get_batch(
                    dataset=val_data, batch_size=cfg.batch_size, context_length=cfg.context_length, device=device
                )
                y_pred = model(x_val)
                val_loss = cross_entropy(pred=y_pred, targets=y_val)
                # wandb.log({"Validation loss": val_loss})
                # sample from the model
                tokenizer = Tokenizer.from_files(
                    vocab_filepath="cs336_basics/tokenizer_data/vocab-tiny.pkl",
                    merges_filepath="cs336_basics/tokenizer_data/merges-tiny.pkl",
                )
                # model.sample(tokenizer=tokenizer, prompt="It was a nice day")

        if step % cfg.log_interval == 0:
            print(loss)
            # wandb.log({"Loss": loss})

        if step % cfg.save_interval == 0:
            save_checkpoint(model=model, optimizer=optimizer, iteration=step, out=cfg.output_dir)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    # write to output path
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
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

    batch = torch.from_numpy(dataset[indices]).to(device)

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
    for param in parameters:
        if param.grad != None:
            param.grad *= max_l2_norm / (l2_norm + eps)


if __name__ == "__main__":
    base_config = Config
    base_config = Config.from_yaml("cs336_basics/configs/base.yaml")
    print(base_config.batch_size)
    # import sys; sys.exit()
    train(cfg=base_config)
    """
    weights = torch.nn.Parameter(5 * torch.randn((10,10)))
    opt = SGD([weights], lr=1e3)
    for i in range(10):
        opt.zero_grad()
        #calculate loss
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()"""
