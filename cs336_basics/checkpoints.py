import argparse
import glob
import os

import regex as re

import numpy as np
import torch

from cs336_basics.config import Config, get_parser
from cs336_basics.train_model import calculate_loss, get_model, load_checkpoint, save_checkpoint

# Experimenting with using more classes and structuring my code properly (according to Claude)
# This will load multiple checkpoints, merge those and validate the resulting merged model. It will not do anything to the optimizer state.


class Evaluator:
    # Loads checkpoints, merges them, then does full validation
    @staticmethod
    def validate_model_checkpoint(
        checkpoint_path,
        cfg: Config,
    ):
        """Opens the model path and returns n samples from the model"""
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        cfg.dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
        val_data = np.load(cfg.val_dataset_path, mmap_mode="r")

        model = get_model(cfg, device) if device == torch.device("cpu") else torch.compile(get_model(cfg, device))

        load_checkpoint(checkpoint_path, model)
        val_loss = calculate_loss(model, val_data, cfg, 0, device, rng=None, num_iters=150)
        print(f"The validation loss is {val_loss:.4f}")
        return val_loss


class CheckpointManager:
    @staticmethod
    def discover_checkpoints(base_name, granularity, n_checkpoints):
        """Can fail for all kinds of reasons if data is not correctly formatted or stored in the right order"""

        full_list = glob.glob(f"{base_name}_*")
        print(f"Available checkpoints {full_list}")
        checkpoints = []
        prev = None
        for cp in reversed(full_list):  # start from largest checkpoint and go back steps with
            n = int(cp.split(base_name + "_")[-1].strip(".pth"))
            if prev is None or prev == n + granularity:
                prev = n
                checkpoints.append(cp)
        # Validate n_checkpoints ok
        if len(checkpoints) < n_checkpoints:
            return None
        # print(checkpoints)
        return list(reversed(checkpoints[:n_checkpoints]))


class ModelGrower:
    @staticmethod
    def grow_checkpoint(src, size, output):
        #
        grown_model = {}
        grown_optimizer = {}
        # load checkpoint
        if not os.path.exists(src):
            raise FileNotFoundError(f"Checkpoint not found at {src}")
        checkpoint = torch.load(src) if torch.cuda.is_available() else torch.load(src, map_location=torch.device("cpu"))
        # update state
        model_dict = checkpoint["model"]
        opt_dict = checkpoint["optimizer"]
        step = checkpoint["iteration"]
        layer_pattern = re.compile(r'(_orig_mod\.layers\.)(\d+)(\..*)')
        layer_no = None

        # growth logic (stacking two models on top of each other)
        for k, v in model_dict.items():
            match = re.search(layer_pattern, k)
            if match:
                prefix = match.group(1)      # '_orig_mod.layers.'
                layer_num = int(match.group(2))  # 0, 1, 2, ...
                suffix = match.group(3)      # '.mha.Wq.W' etc.
                
                new_key = f"{prefix}{layer_num + size}{suffix}"
                grown_model[new_key] = v
                
            grown_model[k] = v
            layer_no = None

        


        checkpoint = {"model": grown_model, "optimizer": None, "iteration": step}
        print(f"Saving grown checkpoint to {output}")
        torch.save(checkpoint, output)

        return 


class ModelMerger:
    @staticmethod
    def compute_merge_weights(stratergy_name, n_checkpoints):
        if stratergy_name == "linear":
            return [1 / n_checkpoints for _ in range(n_checkpoints)]
        if stratergy_name == "sqrt":
            raise NotImplementedError()

    @staticmethod
    def merge_checkpoints(checkpoint_paths, weights, output):
        # go through checkpoints one by one to build up a merged model state
        merged_state = {}
        assert len(checkpoint_paths) == len(weights)  # Must have same amount of weights and checkpoints
        for src, weight in zip(checkpoint_paths, weights):
            # load checkpoint
            if not os.path.exists(src):
                raise FileNotFoundError(f"Checkpoint not found at {src}")
            checkpoint = (
                torch.load(src) if torch.cuda.is_available() else torch.load(src, map_location=torch.device("cpu"))
            )
            # update state
            state_dict = checkpoint["model"]
            if not merged_state:
                merged_state = {k: v * weight for k, v in state_dict.items()}
            else:
                for k, v in state_dict.items():
                    merged_state[k] += v * weight

        checkpoint = {"model": merged_state, "optimizer": {}, "iteration": -1}
        print(f"Saving merged checkpoint to {output}")
        torch.save(checkpoint, output)

        return


class ExperimentRunner:
    @staticmethod
    def run_experiment(base_name: str, n_checkpoints: int, granularity: int, stratergy: str, cfg):
        """Single experiment"""
        output_path = f"{base_name}-merged_{stratergy}_{n_checkpoints}_{granularity}.pth"
        checkpoint_paths = CheckpointManager.discover_checkpoints(base_name, granularity, n_checkpoints)
        if checkpoint_paths is None:
            print(f"No checkpoints found with base name {base_name}")
            return None

        merge_weights = ModelMerger.compute_merge_weights(stratergy, n_checkpoints)
        ModelMerger.merge_checkpoints(checkpoint_paths, merge_weights, output_path)
        loss = Evaluator.validate_model_checkpoint(output_path, cfg)
        return loss

    @staticmethod
    def run_sweep(base_name, checkpoint_ranges, stratergies, granularities, config):
        # runs multiple merge_checkpoints to test various alternatives
        results = []
        for num_checkpoints in checkpoint_ranges:
            for granularity in granularities:
                for stratergy in stratergies:
                    loss = ExperimentRunner.run_experiment(
                        base_name=base_name,
                        n_checkpoints=num_checkpoints,
                        granularity=granularity,
                        stratergy=stratergy,
                        config=config,
                    )
                    results.append([num_checkpoints, stratergy, granularity, loss])

        for result in results:
            print(result)


if __name__ == "__main__":
    # test_merging_stratergy("wsm", [8, 12, 16, 20], ["linear"]) # for running a sweep of merging stratergies

    parser = get_parser()
    args = parser.parse_args()

    # load config
    config = Config.from_yaml(args.config)
    config.update_from_args(args)

    """ModelGrower.grow_checkpoint(
        src="cs336_basics/configs/experiments/6_layers_deep_4800.pth", size = 6, output="cs336_basics/configs/experiments/12_layers_deep_cp.pth",
    )
    import sys; sys.exit();"""
    ExperimentRunner.run_experiment(
        base_name="cs336_basics/configs/experiments/6_layers_deep",
        n_checkpoints=10,
        granularity=200,
        stratergy="linear",
        cfg=config,
    )
    # uv run cs336_basics/train_model.py --config cs336_basics/configs/stack.yaml --num_layers 12 --from_checkpoint 12_layers_deep_250.pth