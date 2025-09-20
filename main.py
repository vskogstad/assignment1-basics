import os

from cs336_basics.config import Config
from cs336_basics.train_model import train
from cs336_basics.utils import download_training_file

if __name__ == "__main__":
    # load config
    config = Config.from_yaml("cs336_basics/configs/best.yaml")

    # Download training_data files if needed.
    files = {"data/training_data/owt_train.npy": "https://huggingface.co/datasets/vskogstad/OpenWebText-train/resolve/main/owt_train.npy", 
             "data/training_data/owt_valid.npy": "https://huggingface.co/datasets/vskogstad/OpenWebText-train/resolve/main/owt_valid.npy"}
    for file, link in files.items():
        if not os.path.exists(file):
            print(f"no {file}")
            download_training_file(file, link)
    train(cfg=config)
    # Save final config to experiment directory
    os.makedirs(config.output_dir, exist_ok=True)
    config.save(os.path.join(config.output_dir, f"{config.experiment_name}_config.yaml"))
