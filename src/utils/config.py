import yaml
import os
from pathlib import Path

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def ensure_dirs(config: dict):
    dirs = [
        config["data"]["raw_dir"],
        config["data"]["processed_dir"],
        config["outputs"]["checkpoint_dir"],
        config["outputs"]["results_dir"],
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ All directories ready.")