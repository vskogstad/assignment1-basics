from cs336_basics.train_model import train
from cs336_basics.config import Config


if __name__ == "__main__":

    # load config
    config = Config.from_yaml("cs336_basics/configs/sweep_lr_24.yaml")
    train(cfg=config) 