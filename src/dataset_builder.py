"""Module for building dataset structure and saving metadata, using DatasetManager."""
import os
import json
import hashlib
from typing import Dict, Any, Optional
import threading
from .dataset_manager import DatasetManager

class DatasetBuilder:
    """
    Handles the creation of dataset files and metadata, working with DatasetManager
    for organization and management.
    """
    def __init__(self, dataset_name: str, output_base_dir: str):
        """
        Initializes the DatasetBuilder with DatasetManager integration.

        Args:
            dataset_name: The name for the dataset.
            output_base_dir: The base directory where datasets will be organized.
        """
        self.dataset_name = dataset_name
        self.manager = DatasetManager(output_base_dir)
        
        # Initialize dataset directory structure using manager
        self.dataset_dir, self.images_dir = self.manager.init_dataset(dataset_name)
        self.metadata_path = self.manager.create_metadata_file(self.dataset_dir)
        
        self.image_count = 0
        self._file_lock = threading.Lock()  # Lock for thread-safe file writing
        print(f"Dataset initialized at: {self.dataset_dir}")

    def _generate_filename(self, image_bytes: bytes, metadata: Dict[str, Any]) -> str:
        """
        Generates a unique filename for the image.
        Uses a hash of image bytes and key metadata for uniqueness and reproducibility.

        Args:
            image_bytes: The image data in bytes.
            metadata: The metadata associated with the image.

        Returns:
            A unique filename string (e.g., "img_s123_cfg7_prompt_abc123def.png").
        """
        self.image_count += 1
        base_name = f"image_{self.image_count:05d}"  # Default sequential name

        try:
            # Extract key parameters safely using .get()
            seed = metadata.get('request_parameters', {}).get('seed', 'unknown')
            cfg = metadata.get('request_parameters', {}).get('cfg_scale', 'unknown')
            # Use first few words of prompt (cleaned)
            prompt_str = str(metadata.get('request_parameters', {}).get('prompt', ''))
            prompt_prefix = "_".join(prompt_str.split()[:3]).lower()
            prompt_prefix = ''.join(c for c in prompt_prefix if c.isalnum() or c == '_')

            # Add a short hash of image bytes for uniqueness
            img_hash = hashlib.sha1(image_bytes).hexdigest()[:8]

            # Combine parts, ensuring they are strings
            parts = [
                f"img",
                f"s{seed}",
                f"cfg{cfg}",
                f"{prompt_prefix}",
                f"{img_hash}"
            ]
            base_name = "_".join(filter(None, parts))  # Filter out empty parts

        except Exception as e:
            print(f"Warning: Could not generate descriptive filename, using sequential. Error: {e}")

        return f"{base_name}.png"

    def add_image(self, image_bytes: bytes, metadata: Dict[str, Any]):
        """
        Saves an image to the dataset and appends its metadata.

        Args:
            image_bytes: The image data in bytes.
            metadata: The metadata dictionary for this image.
                     Should include A1111 'info' and 'request_parameters'.
        """
        if not image_bytes:
            print("Warning: Received empty image bytes. Skipping.")
            return

        filename = self._generate_filename(image_bytes, metadata)
        filepath = os.path.join(self.images_dir, filename)

        try:
            # Save image
            with open(filepath, "wb") as f:
                f.write(image_bytes)

            # Update metadata with file information
            metadata['filename'] = filename
            metadata['filepath'] = os.path.relpath(filepath, self.dataset_dir)

            # Write metadata line
            self._append_metadata(metadata)

        except IOError as e:
            print(f"Error saving image {filename} or its metadata: {e}")
        except Exception as e:
            print(f"Unexpected error while adding image {filename}: {e}")

    def _append_metadata(self, metadata: Dict[str, Any]):
        """
        Appends a single metadata record to the JSON Lines file.
        Uses file lock for thread safety.
        """
        try:
            with self._file_lock:
                with open(self.metadata_path, 'a', encoding='utf-8') as f:
                    json_record = json.dumps(metadata, ensure_ascii=False)
                    f.write(json_record + '\n')
        except IOError as e:
            print(f"Error appending metadata to {self.metadata_path}: {e}")
        except Exception as e:
            print(f"Unexpected error while appending metadata: {e}")

    def finalize_dataset(self) -> Optional[str]:
        """
        Finalizes the dataset creation process.
        
        Returns:
            The path to the metadata file if images were added, otherwise None.
        """
        if self.image_count == 0:
            print("No images were added to the dataset.")
            # Consider archiving or cleaning up empty dataset
            return None

        # Get dataset info using manager
        try:
            info = self.manager.get_dataset_info(self.dataset_name)
            print(f"Dataset generation finished:")
            print(f"- Total images: {info['image_count']}")
            print(f"- Total size: {info['total_size']} bytes")
            print(f"- Created: {info['created']}")
            print(f"- Metadata: {self.metadata_path}")
            return self.metadata_path
        except Exception as e:
            print(f"Error getting dataset info: {e}")
            print(f"Dataset generation finished. {self.image_count} images saved.")
            print(f"Metadata path: {self.metadata_path}")
            return self.metadata_path

if __name__ == '__main__':
    # Example usage and testing
    print("Testing DatasetBuilder with DatasetManager integration...")
    test_builder = DatasetBuilder("builder_test_managed", "./output_datasets")

    # Create some dummy data
    dummy_image_1 = b"fake_managed_image_data_1"
    dummy_meta_1 = {
        "prompt": "test prompt 1",
        "seed": 111,
        "steps": 10,
        "request_parameters": {"prompt": "test prompt 1", "seed": 111, "cfg_scale": 5}
    }
    dummy_image_2 = b"fake_managed_image_data_2"
    dummy_meta_2 = {
        "prompt": "test prompt 2",
        "seed": 222,
        "steps": 15,
        "request_parameters": {"prompt": "test prompt 2", "seed": 222, "cfg_scale": 7}
    }

    print("\nAdding test images...")
    test_builder.add_image(dummy_image_1, dummy_meta_1)
    test_builder.add_image(dummy_image_2, dummy_meta_2)

    print("\nFinalizing dataset...")
    final_meta_path = test_builder.finalize_dataset()

    if final_meta_path and os.path.exists(final_meta_path):
        print("\nTest successful!")
        # Export dataset as a demonstration
        try:
            zip_path = test_builder.manager.export_dataset("builder_test_managed")
            print(f"Exported dataset to: {zip_path}")
        except Exception as e:
            print(f"Error during export: {e}")
    else:
        print("\nTest failed.")
