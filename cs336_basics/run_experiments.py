import subprocess
import sys


experiments = [
    
    #("cs336_basics/configs/sweep_lr9.yaml", "Lr9"),
    #("cs336_basics/configs/sweep_lr10.yaml", "Lr10"),
    #("cs336_basics/configs/sweep_lr18.yaml", "Lr18"),
    ("cs336_basics/configs/sweep_lr24.yaml", "Lr24"),
    # Add more experiments here...
]

def run_experiment(config_file, experiment_name):
    cmd = [
        "uv", "run", "cs336_basics/train_model.py",
        "--config", config_file,
        "--experiment_name", experiment_name
    ]
    
    print(f"Starting experiment: {experiment_name}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"Completed: {experiment_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed: {experiment_name} (exit code: {e.returncode})")
        return False
    
    return True

def main():
    print(f"Running {len(experiments)} experiments")
    
    for i, (config_file, experiment_name) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Running experiment...")
        
        success = run_experiment(config_file, experiment_name)
        
        if not success:
            print(f"\nStopping due to failed experiment: {experiment_name}")
            sys.exit(1)
    
    print(f"\n All experiments completed.")

if __name__ == "__main__":
    main()