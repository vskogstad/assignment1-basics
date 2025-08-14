from cs336_basics.train_model import train
from cs336_basics.config import Config


if __name__ == "__main__":

    # load config
    config = Config.from_yaml("cs336_basics/configs/base_owt.yaml")
    train(cfg=config) 