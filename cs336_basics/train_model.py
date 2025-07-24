import math

import torch
from einops import einsum, rearrange

from torch import nn as nn
from collections.abc import Callable, Iterable
from typing import Optional


@staticmethod
def softmax(x: torch.Tensor, dimension: int):
    max_x = torch.max(x, dim=1, keepdim=True).values
    print(torch.max(x, dim=1))
    result = torch.exp(x-max_x) / torch.sum(torch.exp(x-max_x), dim=dimension, keepdim=True)

    return result


def cross_entropy(x: torch.Tensor, targets: torch.Tensor):
    """Calculates the negative log likelyhood"""
    batch_size = x.size(0)
    max_x = torch.max(x, dim=1, keepdim=True).values

    x_shifted = x - max_x # Compute log-sum-exp: log(sum(exp(x_i)))
    log_sum_exp = torch.log(torch.sum(torch.exp(x_shifted), dim=1, keepdim=True))
    
    # Get the logits for the target classes
    target_logits = x_shifted[torch.arange(batch_size), targets].unsqueeze(1)
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
    def __init__(self, params, lr=1e-3, betas = (0.9, 0.999), weight_decay = 1e-4, eps = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr,
                    "betas": betas,
                    "weight_decay": weight_decay,
                    "eps": eps}
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
                m = state.get("m", torch.zeros_like(p.data))  # Get first vector momentum from the state, or initial value.
                v = state.get("v", torch.zeros_like(p.data))  # Get second vector momentum from the state, or initial value.
                
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                m = beta_1 * m + (1 - beta_1) * grad    # Update first moment estimate
                v = beta_2 * v + (1 - beta_2) * grad**2 # Update second moment estimate

                lr_t = lr * (math.sqrt(1 - beta_2**t) / (1 - beta_1**t))  # adjust lr
                p.data -= lr_t * (m / (torch.sqrt(v) + eps))  # Update weight tensor in-place.
                p.data -= lr * weight_decay * p.data # Apply weight decay
                state["t"] = t + 1  # Increment iteration number.
                state["m"] = m
                state["v"] = v
        
        return loss
    
if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10,10)))
    opt = SGD([weights], lr=1e3)
    for i in range(10):
        opt.zero_grad()
        #calculate loss
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
