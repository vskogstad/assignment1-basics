import argparse
import math
import os
import timeit
from collections.abc import Callable, Iterable
from dataclasses import asdict
from typing import IO, BinaryIO, Optional

import numpy as np
import torch
from einops import einsum, rearrange
from torch import nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

import wandb
from cs336_basics.config import Config, get_parser
from cs336_basics.model import Transformer
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.utils import resource_accounting, step_law_lr

# Done
# x Test the optimization with pytorch matrix-sizes. (options max_autotune) (MFU A bit higher, but don't make up the time spent optimizing)
# x Test scaling ln with depth     https://arxiv.org/pdf/2502.05795
# x Test corrected optimal batch-size and learning rate.
# x Grouped query attention.
# x Gated attention
# x Scaling with sqrt(2 * depth) like in Ernie
# x sliding window attention
# TODO
# Perplexity measurement.


def train(cfg: Config):
    """Main training loop
    Initializes a training run based on parameters in config and cmd-line arguments. --config is a required cmd-line argument
    """
    t_start = t_0 = timeit.default_timer()

    # Reproducability.
    # Note that the regular random module is used for sampling responses from the model. Not seeded cause I like seeing new text.
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    torch.set_float32_matmul_precision("high")  # Improved speed
    # torch.autograd.set_detect_anomaly(True) # for bughunting
    # torch._logging.set_logs(graph_breaks=True) # for graph-breaks

    if torch.cuda.is_available():
        # Clear GPU cache between runs
        torch.cuda.empty_cache()
        # Set CUDA memory allocation strategy (Neccessary to avoid OOM issues)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

        if cfg.dtype == "bfloat16":
            max_flops = 989e12  # BF16 Tensor Core without sparsity.
        else:
            max_flops = 989e12 / 2  # TF32 Tensor Core without sparsity.
    else:
        max_flops = float("inf")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cfg.dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32

    # torch.set_default_dtype(dtype)
    print(f"Using {device} and {cfg.dtype}")
    print(f"Processing {cfg.batch_size * cfg.context_length * cfg.total_steps:,} tokens")
    flops_per_batch, non_embedding_params = resource_accounting(cfg)  # Print info about the current run to the log
    step_law_lr(
        len_data=140_000 * 5200, non_embedding_params=non_embedding_params, context_length=cfg.context_length
    )  # ~140 000 tokens/sec, 5200 secs available (200 for loading model and compilation)

    # Load model, optimizer and scheduler
    model = (
        get_model(cfg, device) if device == torch.device("cpu") else torch.compile(get_model(cfg, device))
    ) 
    optimizer = get_optimizer(cfg=cfg, model=model)

    assert cfg.scheduler == "cosine" or cfg.scheduler == "wsd"  # Not setup to use other schedulers yet.
    loss_func = torch.compile(cross_entropy, fullgraph=True)

    # Data loading
    train_data = np.load(cfg.train_dataset_path, mmap_mode="r")
    print(f"The training dataset contains {len(train_data):,} tokens")
    val_data = np.load(cfg.val_dataset_path, mmap_mode="r")
    """tokenizer = Tokenizer.from_files(
        vocab_filepath=cfg.tokenizer_vocab_path,
        merges_filepath=cfg.tokenizer_merges_path,
    )"""
    # We use randomized unrepeated data if we have enough available, break out if not
    assert(len(train_data) // (cfg.context_length * cfg.batch_size) > cfg.total_steps)
    
    # TODO: Multi-epoch training not implemented
    max_idx = len(train_data) // (cfg.context_length)
    starts = np.arange(max_idx) * cfg.context_length
    rng.shuffle(starts)
    starts = starts[:cfg.total_steps * cfg.batch_size].reshape(-1, cfg.batch_size)

    # Initialize logging
    if cfg.wandb_project:
        wandb.login()
        run = wandb.init(project=cfg.wandb_project, config=asdict(cfg))
    current_step = 0
    current_tokens = 0
    chunk_steps = 0
    chunk_loss = 0

    # Load from checkpoint
    if cfg.from_checkpoint:
        source = os.path.join(cfg.output_dir, cfg.from_checkpoint)
        step = (
            load_checkpoint(src=source, model=model, optimizer=optimizer) + 1
        )  # increment by one, this step has already been done
    
    
    
    for step in range(current_step, cfg.total_steps):
        optimizer.zero_grad()
        x, y = get_batch_nonrepeating(
                    dataset=train_data, batch_size=cfg.batch_size,
                    context_length=cfg.context_length, device=device,
                    starts=starts[step],
                    
                )
        if cfg.sky_ladder:
            w = min(cfg.context_length, max(cfg.sky_ladder_min_window, (cfg.sky_ladder_step_interval * ((cfg.sky_ladder_alpha*step)//cfg.sky_ladder_step_interval))))
            seq_windows = torch.cat([
                torch.arange(0, cfg.context_length, w, device=device, dtype=torch.int32),
                torch.tensor([cfg.context_length], device=device, dtype=torch.int32)
            ])
        else: seq_windows = None

        if cfg.grad_accum_steps == 1: # Faster than loop
            with torch.autocast(device_type="cuda", dtype=cfg.dtype):
                y_pred = model(x, seq_windows)
                loss = loss_func(pred=y_pred, targets=y)

            loss_accum = loss.detach()
            loss.backward()

        else:
            x_s = torch.split(x, cfg.batch_size//cfg.grad_accum_steps, dim=0)
            y_s = torch.split(y, cfg.batch_size//cfg.grad_accum_steps, dim=0)
            loss_accum = 0
            for mini_step in range(cfg.grad_accum_steps):
                x = x_s[mini_step]
                y = y_s[mini_step]
                with torch.autocast(device_type="cuda", dtype=cfg.dtype):
                    y_pred = model(x)
                    loss = loss_func(pred=y_pred, targets=y)/cfg.grad_accum_steps
                    loss_accum += loss.detach()
                    loss.backward()


        clip_gradient(parameters=model.parameters(), max_l2_norm=cfg.grad_clip_norm)
        
        if cfg.scheduler == "cosine":
            new_lr = get_lr_cosine(
                step=step,
                max_learning_rate=cfg.max_learning_rate,
                min_learning_rate=cfg.min_learning_rate,
                warmup_steps=cfg.warmup_steps,
                cosine_cycle_steps=cfg.cosine_cycle_steps,
            )
        elif cfg.scheduler == "wsd":
            new_lr = get_lr_wsd(
                step=step,
                max_learning_rate=cfg.max_learning_rate,
                min_learning_rate=cfg.min_learning_rate,
                warmup_steps=cfg.warmup_steps,
                steady_steps=cfg.steady_steps,
                total_steps=cfg.total_steps,
            )
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

        optimizer.step()


        # Logging
        chunk_loss += loss_accum
        chunk_steps += 1
        current_tokens += cfg.batch_size * cfg.context_length
        if step % cfg.log_interval == 0:
            loss_accumulated = chunk_loss.sum().item() / chunk_steps
            if device == torch.device("cuda"):
                torch.cuda.synchronize()
            t_1 = timeit.default_timer()
            chunk_time = t_1 - t_0
            t_0 = timeit.default_timer()


            token_per_s = chunk_steps * cfg.batch_size * cfg.context_length / (chunk_time)
            mfu = (flops_per_batch * cfg.log_interval / (chunk_time)) / max_flops if max_flops != float("inf") else None
            print(
                f"step = {step} | Loss = {loss_accumulated:.3f} | {chunk_time = :.2f} | tok/s = {token_per_s:,.1f} | lr = {new_lr:.5f} | MFU = {mfu:.3f}"
            )
            

            chunk_loss = 0
            chunk_steps = 0
            if cfg.wandb_project:
                run.log(
                    data={
                        "Loss": loss_accumulated,
                        "tok/ms": token_per_s,
                        "lr": new_lr,
                        "MFU": mfu,
                        "Step": step,
                        "time": timeit.default_timer() - t_start,
                    },
                    step=current_tokens,
                )

        # validation
        if step % cfg.eval_interval == 0 and step != 0:
            print(f"skip: {model.skip}, lambdas: {model.lambdas}")
            val_loss = calculate_loss(model, val_data, cfg, current_tokens, device, rng=rng, num_iters=15)
            print(f"The validation loss is {val_loss:.4f}")
            if cfg.wandb_project:
                wandb.log({"Validation loss": val_loss}, step=current_tokens)
            # print(f"Sampling from the model at step {step}")
            # model.sample(tokenizer=tokenizer, prompt="It was a nice day")

        # checkpointing
        if step % cfg.save_interval == 0 and step >= cfg.initial_save_step:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step,
                out=f"{cfg.output_dir}/{cfg.experiment_name}_{step}.pth",
            )

    if cfg.wandb_project:
        run.finish()
    print(f"Total time {timeit.default_timer() - t_start:3f} seconds spent.")
    
    # Do full validation after training model
    val_loss = calculate_loss(model, val_data, cfg, current_tokens, device, rng=rng)
    print(f"The full validation loss is {val_loss:.4f}")



def calculate_loss(model, data, cfg, current_tokens, device, rng=None, num_iters = None):
    model.eval()
    accum_loss = 0
    if not num_iters:  # Use full validation set unless specified
        num_iters = math.ceil(len(data) / (cfg.context_length * cfg.batch_size))
    # print(f"{num_iters=}")
    for i in range(num_iters):
        # print(f"This is {i=}")
        with torch.no_grad():
            x_val, y_val = get_batch(
                dataset=data,
                batch_size=cfg.batch_size,
                context_length=cfg.context_length,
                device=device,
                rng=None,
                current_iter=i,
            )
            
            if cfg.grad_accum_steps == 1: # Faster than loop
                with torch.autocast(device_type="cuda", dtype=cfg.dtype):
                    y_pred = model(x_val, None)
                    # print(f"We get here for {i}")
                    val_loss = cross_entropy(pred=y_pred, targets=y_val)
                    accum_loss += val_loss.item()

            else:
                x_s = torch.split(x_val, cfg.batch_size//cfg.grad_accum_steps, dim=0)
                y_s = torch.split(y_val, cfg.batch_size//cfg.grad_accum_steps, dim=0)
                loss_accum = 0
                for mini_step in range(cfg.grad_accum_steps):
                    x = x_s[mini_step]
                    y = y_s[mini_step]
                    with torch.autocast(device_type="cuda", dtype=cfg.dtype):
                        y_pred = model(x, None)
                        loss = cross_entropy(pred=y_pred, targets=y)/cfg.grad_accum_steps
                        accum_loss += loss.detach()




            
    torch.cuda.empty_cache()
    model.train()
    return accum_loss / num_iters


def monitor_norms(model, step, y_pred):
    print(f"\n=== Step {step} Weight Analysis ===")
    # Embedding weights
    emb_std = model.embedding.embedding.std().item()
    emb_max = model.embedding.embedding.abs().max().item()
    print(f"Embedding: std={emb_std:.4f}, max={emb_max:.4f}")

    # Attention weights (first few layers)
    for i in range(min(3, len(model.layers))):
        layer = model.layers[i]
        # Check attention projection weights
        for name, param in layer.mha.named_parameters():
            if "weight" in name or "W" in name:
                std = param.std().item()
                max_val = param.abs().max().item()

                print(f"Layer {i} MHA {name}: std={std:.4f}, max={max_val:.4f}")

        # FFN weights
        for name, param in layer.ffn.named_parameters():
            if "w" in name:
                std = param.std().item()
                max_val = param.abs().max().item()
                print(f"Layer {i} FFN {name}: std={std:.4f}, max={max_val:.4f}")

    # Output head
    head_std = model.lm_head.W.std().item()
    head_max = model.lm_head.W.abs().max().item()
    print(f"LM Head: std={head_std:.4f}, max={head_max:.4f}")

    # Current output variance
    print(f"Y_pred range: [{y_pred.min().item():.3f}, {y_pred.max().item():.3f}]")

    # Calculate l2_norm
    norm_squared = 0
    for name, param in model.named_parameters():
        if param.grad != None:
            norm_squared += torch.linalg.vector_norm(param.grad) ** 2
    l2_norm = math.sqrt(norm_squared)
    print(f"L2 norm {l2_norm:.4f}")


def get_optimizer(cfg: Config, model):
    """Returns an initialized optimizer with parameters from config"""
    # Currently only supports AdamW, but setup of parameters to also could initialize Muon optimizer later

    hidden_matrix_params = [
        param for name, param in model.named_parameters()  
        if param.ndim >= 2 and "embed" not in name and "lm_head" not in name
    ]
    embed_params = [param for name, param in model.named_parameters() if "embed" in name]
    scalar_params = [param for param in model.parameters() if param.ndim < 2]
    head_params = [model.lm_head.W]

    if cfg.optimizer == "adamw":
        optimizer = AdamW(
            [
                {"params": hidden_matrix_params, "weight_decay": cfg.weight_decay},
                {"params": embed_params, "weight_decay": 0},
                {"params": scalar_params, "weight_decay": cfg.weight_decay},
                {"params": head_params, "weight_decay": cfg.weight_decay},
            ],
            lr=cfg.min_learning_rate,
            betas=cfg.betas,
            eps=cfg.eps,
        )
    elif cfg.optimizer == "muon":
        optimizer = MuonWithAdamW(
            [
                {"params": hidden_matrix_params, "weight_decay": cfg.weight_decay, "use_muon": True, "lr_scale": 1},
                {"params": embed_params, "weight_decay": 0, "use_muon": False, "lr_scale": 1},
                {"params": scalar_params, "weight_decay": 0, "use_muon": False, "lr_scale": 1},
                {"params": head_params, "weight_decay": cfg.weight_decay/3, "use_muon": False, "lr_scale": 1/3},
            ],
            # lr=cfg.max_learning_rate,
            momentum=cfg.muon_momentum,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            eps=cfg.eps,
        )
    elif cfg.optimizer == "normuon":
        optimizer = NorMuonWithAdamW(
            [
                {"params": hidden_matrix_params, "weight_decay": cfg.weight_decay, "use_muon": True, "lr_scale": 1},
                {"params": embed_params, "weight_decay": 0, "use_muon": False, "lr_scale": 1},
                {"params": scalar_params, "weight_decay": 0, "use_muon": False, "lr_scale": 1},
                {"params": head_params, "weight_decay": cfg.weight_decay/3, "use_muon": False, "lr_scale": 1/3},
            ],
            # lr=cfg.max_learning_rate,
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
            dtype=torch.float32, #hard-coded 
            pre_norm=cfg.pre_norm,
            layer_norm=cfg.layer_norm,
            glu=cfg.glu,
        )
    else:
        raise NotImplementedError("Not implemented loading for model {cfg.model_name}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    if optimizer:
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    # write to output path
    else:
        checkpoint = {"model": model.state_dict(), "optimizer": {}, "iteration": iteration}
    print(f"Saving checkpoint to {out}")
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
):
    if not os.path.exists(src):
        raise FileNotFoundError(f"Checkpoint not found at {src}")
    if torch.cuda.is_available():
        checkpoint = torch.load(src)
    else:
        checkpoint = torch.load(src, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Loading checkpoint from {src}")
    else:
        print(f"Loading model from {src} for sampling/validation only.")
    return checkpoint["iteration"]



def get_batch_nonrepeating(
    dataset: np.array, batch_size: int, context_length: int, device: str, starts, rng=None, current_iter: int | None = None
):
    """
    Gets a random or fixed batch of x and y tensors for the model to train on. If a rng is passed, it will be used.
    If current_iter is passed the function will return the current_iter batch in the data_set.
    """
    assert rng is None or current_iter is None  # we should never be passing both a generator and a fixed starting point

    if len(dataset) < context_length:
        raise ValueError("Dataset smaller than context length, not possible to create batches")

    # Generate random starting indices unless we get passed a generator
    max_start_idx = len(dataset) - context_length - 1

    if rng:
        starts = rng.integers(0, max_start_idx + 1, size=batch_size)
    elif current_iter is not None:
        if len(dataset) < current_iter * context_length * batch_size:
            raise ValueError("Trying to access a chunk of the dataset that does not exist.")
        # We want to test for the entire dataset, starting pos.
        # Eeach batch has "batch_size" number of samples with length = "context_length"
        # we want to make sure each starting point are context_length apart
        offset = current_iter * context_length * batch_size  # scale offset by previous batches
        # print(f"{starting_pos = } | {batch_size = } | {context_length = } | {offset = }")
        starts = np.arange(batch_size) * context_length + offset
        # print(starts)
        while starts[-1] > max_start_idx:  # On final section, decrease batch_size to fit remaining data.
            starts = starts[:-1]
    else:
        starts = starts # np.random.randint(0, max_start_idx + 1, size=batch_size)

    # Create index arrays for vectorized sampling
    indices = starts[:, None] + np.arange(context_length + 1)

    batch = torch.from_numpy(dataset[indices]).to(device=device, dtype=torch.int32)  # .dtype(torch.int16)

    x = batch[:, :-1]
    y = batch[:, 1:]
    # print(x[0][0])
    return x, y

def get_batch(
    dataset: np.array, batch_size: int, context_length: int, device: str, rng=None, current_iter: int | None = None
):
    """
    Gets a random or fixed batch of x and y tensors for the model to train on. If a rng is passed, it will be used.
    If current_iter is passed the function will return the current_iter batch in the data_set.
    """
    assert rng is None or current_iter is None  # we should never be passing both a generator and a fixed starting point

    if len(dataset) < context_length:
        raise ValueError("Dataset smaller than context length, not possible to create batches")

    # Generate random starting indices unless we get passed a generator
    max_start_idx = len(dataset) - context_length - 1

    if rng:
        starts = rng.integers(0, max_start_idx + 1, size=batch_size)
    elif current_iter is not None:
        if len(dataset) < current_iter * context_length * batch_size:
            raise ValueError("Trying to access a chunk of the dataset that does not exist.")
        # We want to test for the entire dataset, starting pos.
        # Eeach batch has "batch_size" number of samples with length = "context_length"
        # we want to make sure each starting point are context_length apart
        offset = current_iter * context_length * batch_size  # scale offset by previous batches
        # print(f"{starting_pos = } | {batch_size = } | {context_length = } | {offset = }")
        starts = np.arange(batch_size) * context_length + offset
        # print(starts)
        while starts[-1] > max_start_idx:  # On final section, decrease batch_size to fit remaining data.
            starts = starts[:-1]
    else:
        starts = np.random.randint(0, max_start_idx + 1, size=batch_size)

    # Create index arrays for vectorized sampling
    indices = starts[:, None] + np.arange(context_length + 1)

    batch = torch.from_numpy(dataset[indices]).to(device=device, dtype=torch.int32)  # .dtype(torch.int16)

    x = batch[:, :-1]
    y = batch[:, 1:]
    # print(x[0][0])
    return x, y


def softmax(x: torch.Tensor, dimension: int):
    max_x = torch.max(x, dim=1, keepdim=True).values
    print(torch.max(x, dim=1))
    result = torch.exp(x - max_x) / torch.sum(torch.exp(x - max_x), dim=dimension, keepdim=True)

    return result


def cross_entropy(pred: torch.Tensor, targets: torch.Tensor):
    """Calculates the negative log likelihood"""
    pred = rearrange(pred, "b ... c -> (b ...) c")  # combine batch sequence dimension if present
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
    """Implementation of the Muon optimizer. Just usable for matrices."""

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
            lr = group["lr"] * group["lr_scale"] # Get the learning rate.
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
    """
    Uses the modified Muon with transfer of LR parameters from adamw using the following formula:
    Wt = Wt−1 - ηt(0.2·Ot · sqrt(max(A,B))+λWt−1)
    η = Learning rate
    λ = weight decay
    A,B is the matrix dimensions of the parameter

    https://arxiv.org/abs/2502.16982
    """

    def __init__(self, params, lr=1e-3, momentum=0.95, betas=(0.9, 0.95), weight_decay=1e-4, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "eps": eps, "momentum": momentum, "weight_decay": weight_decay, "betas": betas}
        super().__init__(params, defaults)


    @torch.no_grad()
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

                if not group["use_muon"]:  # Use adamw
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

                    
                    corr_1 = 1 - beta_1**t
                    corr_2 = math.sqrt(1 - beta_2**t)

                    lr_t = lr * (corr_2 / corr_1)  # adjust lr
                    denominator = (torch.sqrt(v) + eps * corr_2) # https://x.com/Tim_Dettmers/status/1969131103798567295
                    p.data = p.data - lr_t * (m / denominator)  # Update weight tensor in-place.
                    p.data = p.data - lr * weight_decay * p.data  # Apply weight decay
                    state["t"] = t + 1  # Increment iteration number.
                    state["m"] = m
                    state["v"] = v

                else:  # Use Muon
                    state = self.state[p]  # Get state associated with p.
                    B_t = state.get("B_t", torch.zeros_like(p.data))

                    grad = p.grad.data  # Get the gradient of loss with respect to p.
                    B_t = momentum * B_t + (1 - momentum) * grad
                    update = momentum * B_t + (1 - momentum) * grad
                    O_t = newtonschulz5(update.bfloat16(), steps=5, eps=1e-7)  # Approximate O_t using newton schulz

                    a_dim, b_dim = p.data.shape  # Finding the dimensions of the matrix to scale the learning rate
                    p.data = (
                        p.data - lr * 0.2 * O_t * math.sqrt(max(a_dim, b_dim)) - (lr * weight_decay * p.data)
                    )  # Update weight tensor in-place and do weight decay. Scaled so we can use optimized parameters for adamw.
                    # p.data -= lr * (O_t + weight_decay * p.data)  # Update weight tensor in-place and do weight decay.
                    # Wt = Wt−1 - ηt(0.2·Ot · sqrt(max(A,B))+λWt−1)
                    state["B_t"] = B_t

        return loss



class NorMuonWithAdamW(torch.optim.Optimizer):
    """
    Implements NorMuon. This method already uses the modified Muon with transfer of LR parameters from adamw using roughly the following formula as before:
    Wt = Wt−1 - ηt(0.2·Ot · sqrt(max(A,B))+λWt−1)
    Becomes with NorMuon:
    Wt = Wt−1 - ηt(0.2·Ot · sqrt(A*B)/|Ô|f)+λWt−1)

    η = Learning rate
    λ = weight decay
    A,B is the matrix dimensions of the parameter

    Original transfer from Kimi: https://arxiv.org/abs/2502.16982
    NorMuon Paper: https://arxiv.org/pdf/2510.05491

    """

    def __init__(self, params, lr=1e-3, momentum=0.95, betas=(0.9, 0.95), weight_decay=1e-4, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "eps": eps, "momentum": momentum, "weight_decay": weight_decay, "betas": betas}
        super().__init__(params, defaults)


    @torch.no_grad()
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
                eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if not group["use_muon"]:  # Use adamw
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

                    
                    corr_1 = 1 - beta_1**t
                    corr_2 = math.sqrt(1 - beta_2**t)

                    lr_t = lr * (corr_2 / corr_1)  # adjust lr
                    denominator = (torch.sqrt(v) + eps * corr_2) # https://x.com/Tim_Dettmers/status/1969131103798567295
                    p.data = p.data - lr_t * (m / denominator)  # Update weight tensor in-place.
                    #wd_mask = torch.where((m / denominator) * p.data >= 0, 1, 0)

                    p.data = p.data - lr * weight_decay * p.data  # Apply weight decay
                    state["t"] = t + 1  # Increment iteration number.
                    state["m"] = m
                    state["v"] = v

                else:  # Use Muon

                    a_dim, b_dim = p.data.shape  # Finding the dimensions of the matrix to scale the learning rate

                    state = self.state[p]  # Get state associated with p.
                    B_t = state.get("B_t", torch.zeros_like(p.data))
                    v_t = state.get("v_t", torch.zeros(a_dim, device=p.device))

                    grad = p.grad.data  # Get the gradient of loss with respect to p.
                    B_t = momentum * B_t + (1 - momentum) * grad
                    update = momentum * B_t + (1 - momentum) * grad # Works better than the proper algorithm.
                    O_t = newtonschulz5(update.bfloat16(), steps=5, eps=1e-7)  # Approximate O_t using newton schulz
                    v_t = momentum * v_t + (1 - momentum) * O_t.square().mean(dim=1) # Use momentum along the columns
                    V_t = v_t.unsqueeze(1)
                    O_mod = O_t / (torch.sqrt(V_t + eps))

                    #a_dim, b_dim = p.data.shape  # Finding the dimensions of the matrix to scale the learning rate
                    eta_hat = 0.2 * lr * math.sqrt(a_dim * b_dim) / (torch.norm(O_mod, p='fro') + 1e-10)
                    #u = eta_hat * O_mod
                    #base = torch.ones_like(p.data)
                    #wd_mask = torch.where(u * p.data >= 0, 1, 0)
                    #print(CWD)
                    #print(f"{(CWD * p.data)=}")
                    #import sys; sys.exit()
                    p.data = p.data - lr * weight_decay * (p.data) - eta_hat * O_mod # Update weight tensor in-place and do weight decay. Scaled so we can use optimized parameters for adamw.
                    # p.data -= lr * (O_t + weight_decay * p.data)  # Update weight tensor in-place and do weight decay.
                    # Wt = Wt−1 - ηt(0.2·Ot · sqrt(max(A,B))+λWt−1)
                    state["B_t"] = B_t
                    state["v_t"] = v_t

        return loss

@torch.compile()
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

#@torch.compile()
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

def get_lr_wsd(
    step: int, max_learning_rate: float, min_learning_rate: float, warmup_steps: int, steady_steps: int, total_steps: int,
):
    """Returns a learning rate based on cosine annealing"""
    if step < warmup_steps:  # Warmup
        lr = step / warmup_steps * max_learning_rate
    elif step > steady_steps:  # Annealing
        lr = max_learning_rate - (max_learning_rate - min_learning_rate) * ((step - steady_steps) / (total_steps - steady_steps))
    # else do nothing
    else:  # Cosine annealing
        lr = max_learning_rate



    return lr


@torch.compile()
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


def sample_from_model_checkpoint(
    model_path,
    cfg: Config,
    num_samples: int,
    prompt: str | None = None,
    max_tokens: int = 256,
    temp: int = 1,
    top_p: int = 0.5,
):
    """Opens the model path and returns n samples from the model"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cfg.dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32

    tokenizer = Tokenizer.from_files(
        vocab_filepath=cfg.tokenizer_vocab_path,
        merges_filepath=cfg.tokenizer_merges_path,
    )
    model = get_model(cfg, device) if device == torch.device("cpu") else torch.compile(get_model(cfg, device))

    load_checkpoint(model_path, model)

    if num_samples > 128:
        print(f"Maximum 128 samples")
        num_samples = 128
    model.sample(tokenizer, prompt, num_samples, max_tokens, temp, top_p)


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # load config
    config = Config.from_yaml(args.config)
    config.update_from_args(args)
    calculate_loss()
    sample_from_model_checkpoint(
        model_path="cs336_basics/configs/experiments/U-net_learnable_10.pth",
        cfg=config,
        num_samples=10,
        prompt="Once upon a time",
    )
    import sys

    sys.exit()

    # Save final config to experiment directory
    os.makedirs(config.output_dir, exist_ok=True)
    config.save(os.path.join(config.output_dir, f"{config.experiment_name}_config.yaml"))

    # train the model
    train(cfg=config)

    import sys

    sys.exit()
    # Sampling snippet
    sample_from_model_checkpoint(
        model_path="cs336_basics/configs/experiments/U-net_learnable_10.pth",
        cfg=config,
        num_samples=10,
        prompt="Once upon a time",
    )
    import sys

    sys.exit()