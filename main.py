import os

from cs336_basics.config import Config
from cs336_basics.train_model import train
from cs336_basics.utils import download_training_files

if __name__ == "__main__":
    # load config
    config = Config.from_yaml("cs336_basics/configs/sweep_lr_24.yaml")

    # Download training_data files if needed.
    for path in ["data/training_data/owt_train_s.npy", "data/training_data/owt_valid.npy"]:
        if not os.path.exists(path):
            print(f"no {path}")
            import sys

            sys.exit()
            download_training_files(path)
    train(cfg=config)
