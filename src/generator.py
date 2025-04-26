# Module orchestrating the image generation process.
import itertools
import os
from typing import Dict, Any, List, Iterator, Tuple

from .config_loader import load_config
from .a1111_client import A1111Client
from .dataset_builder import DatasetBuilder

def generate_parameter_combinations(base_params: Dict[str, Any], variations: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
    """
    Generates all combinations of parameters based on the variations.

    Args:
        base_params: Dictionary of base parameters.
        variations: Dictionary where keys are parameter names and values are lists of values to iterate over.

    Yields:
        Dictionaries, each representing a unique combination of parameters,
        merged with the base parameters.
    """
    if not variations:
        # If no variations, yield the base parameters once
        yield base_params.copy()
        return

    variation_keys = list(variations.keys())
    variation_values = [variations[key] for key in variation_keys]

    # Create all combinations of the variation values
    for value_combination in itertools.product(*variation_values):
        # Create a new parameter set for this combination
        current_params = base_params.copy()
        variation_dict = dict(zip(variation_keys, value_combination))
        current_params.update(variation_dict) # Override base params with variation values
        yield current_params


def run_generation(config_path: str):
    """
    Loads configuration, generates images based on parameter combinations,
    and builds the dataset.

    Args:
        config_path: Path to the configuration YAML file.
    """
    try:
        config = load_config(config_path)
    except (FileNotFoundError, yaml.YAMLError, ValueError, RuntimeError) as e:
        print(f"Error loading configuration: {e}")
        return

    try:
        client = A1111Client(config['api_url'])
    except (ConnectionError, TimeoutError, RuntimeError) as e:
        print(f"Error initializing A1111 client: {e}")
        return

    builder = DatasetBuilder(
        dataset_name=config['dataset_name'],
        output_base_dir=config['output_base_dir']
    )

    print("\nStarting image generation process...")
    total_combinations = 1
    for values in config['parameter_variations'].values():
        total_combinations *= len(values) if values else 1
    print(f"Total parameter combinations to generate: {total_combinations}")

    generated_count = 0
    failed_count = 0

    # Prepare the base payload by copying base_parameters
    base_payload = config.get('base_parameters', {}).copy()

    # Generate combinations
    param_combinations = generate_parameter_combinations(
        base_payload,
        config['parameter_variations']
    )

    for i, params in enumerate(param_combinations):
        print(f"\n--- Generating image {i+1}/{total_combinations} ---")
        print(f"Parameters: {params}")

        # Construct the final payload for the API call
        # Ensure only valid A1111 API parameters are sent
        # We assume 'params' now holds the complete set for this iteration
        api_payload = params # In this setup, params should already be the full payload

        try:
            image_bytes, generation_info = client.generate_image(api_payload)

            if image_bytes and generation_info:
                # Add the used parameters to the metadata we save
                # The 'info' dict from A1111 might already contain most of this,
                # but we explicitly add the combined params for clarity.
                metadata = generation_info.copy() # Start with info from A1111
                metadata['request_parameters'] = params # Add the parameters we sent

                builder.add_image(image_bytes, metadata)
                generated_count += 1
                print(f"Image {i+1} added to dataset.")
            else:
                failed_count += 1
                print(f"Image {i+1} generation failed.")

        except Exception as e:
            failed_count += 1
            print(f"An unexpected error occurred during generation for parameters {params}: {e}")
            # Decide if you want to stop or continue on error
            # continue

    print("\n--- Generation Complete ---")
    print(f"Successfully generated: {generated_count}")
    print(f"Failed generations: {failed_count}")

    if generated_count > 0:
        try:
            metadata_path = builder.finalize_dataset()
            print(f"Dataset saved to: {builder.dataset_dir}")
            print(f"Metadata saved to: {metadata_path}")
        except Exception as e:
            print(f"Error finalizing dataset: {e}")
    else:
        print("No images were generated, dataset not finalized.")

if __name__ == '__main__':
    # Example of how to run this module directly (for testing)
    # Requires a config file and running A1111 API

    # 1. Create a dummy config file
    dummy_config_content_gen = """
api_url: "http://127.0.0.1:7860" # CHANGE IF YOUR API IS ELSEWHERE
dataset_name: "generator_test_dataset"
output_base_dir: "./output_datasets_test" # Use a test output dir
base_parameters:
  prompt: "a futuristic cityscape at sunset"
  negative_prompt: "worst quality, low quality, text, signature"
  steps: 8       # Low steps for faster testing
  width: 256     # Small size
  height: 256
  sampler_name: "Euler a"
parameter_variations:
  seed: [101, 202]
  cfg_scale: [6, 8]
  prompt: ["a retro wave beach scene", "cyberpunk alleyway"] # Override base prompt
"""
    dummy_config_path_gen = "dummy_generator_config.yaml"
    with open(dummy_config_path_gen, 'w', encoding='utf-8') as f:
        f.write(dummy_config_content_gen)
    print(f"Created dummy config: {dummy_config_path_gen}")

    # 2. Ensure A1111 API is running

    # 3. Run the generation
    print("\nRunning generator with dummy config...")
    try:
        run_generation(dummy_config_path_gen)
    except Exception as e:
        print(f"\nError during generator test run: {e}")
    finally:
        # 4. Clean up dummy file and potentially the output dir
        if os.path.exists(dummy_config_path_gen):
            os.remove(dummy_config_path_gen)
            print(f"\nCleaned up {dummy_config_path_gen}")
        # Optional: Clean up test output directory
        # import shutil
        # test_output_dir = os.path.join("output_datasets_test", "generator_test_dataset")
        # if os.path.exists(test_output_dir):
        #     shutil.rmtree(test_output_dir)
        #     print(f"Cleaned up test output directory: {test_output_dir}")
