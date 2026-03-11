"""Configuration loading from YAML files."""

import yaml
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load config from YAML file. Falls back to default if no path given."""
    if path is None:
        path = DEFAULT_CONFIG_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: str | Path):
    """Save config dict to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
