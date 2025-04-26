# Module for loading and validating configuration files (e.g., YAML).
import yaml
import os
from typing import Dict, Any, List

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads the configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file is not valid YAML.
        ValueError: If the configuration is missing required keys.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while reading {config_path}: {e}")

    if config is None:
        raise ValueError(f"Configuration file {config_path} is empty or invalid.")

    # Basic validation (can be expanded)
    required_keys = ['api_url', 'dataset_name', 'output_base_dir', 'base_parameters', 'parameter_variations']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in configuration: '{key}'")

    # Validate parameter_variations structure (basic check)
    if not isinstance(config.get('parameter_variations'), dict):
         raise ValueError("'parameter_variations' must be a dictionary.")
    for param, values in config['parameter_variations'].items():
        if not isinstance(values, list):
            raise ValueError(f"Values for '{param}' in 'parameter_variations' must be a list.")

    print(f"Configuration loaded successfully from: {config_path}")
    return config

if __name__ == '__main__':
    # Example usage (for testing this module directly)
    # Create a dummy config for testing
    dummy_config_content = """
api_url: "http://127.0.0.1:7860"
dataset_name: "test_dataset"
output_base_dir: "./output_datasets_test"
base_parameters:
  prompt: "test prompt"
  negative_prompt: "test negative"
  steps: 10
parameter_variations:
  seed: [1, 2]
  cfg_scale: [5, 7]
"""
    dummy_path = "dummy_config_test.yaml"
    with open(dummy_path, 'w', encoding='utf-8') as f:
        f.write(dummy_config_content)

    try:
        loaded_conf = load_config(dummy_path)
        print("\nDummy config loaded:")
        import json
        print(json.dumps(loaded_conf, indent=2))
    except Exception as e:
        print(f"\nError loading dummy config: {e}")
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path) # Clean up dummy file
            print(f"\nCleaned up {dummy_path}")
