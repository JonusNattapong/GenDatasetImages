"""
Dataset Manager: Handles organization and management of generated datasets.
"""

import os
import shutil
import json
import yaml
from datetime import datetime
from pathlib import Path
import logging

class DatasetManager:
    def __init__(self, base_dir="output_datasets"):
        """
        Initialize the dataset manager.
        Args:
            base_dir (str): Base directory for all datasets
        """
        self.base_dir = Path(base_dir)
        self.archive_dir = self.base_dir / "archive"
        self.tmp_dir = self.base_dir / "tmp"
        
        # Create necessary directories
        self.base_dir.mkdir(exist_ok=True)
        self.archive_dir.mkdir(exist_ok=True)
        self.tmp_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def init_dataset(self, dataset_name, config=None):
        """
        Initialize a new dataset directory structure.
        Args:
            dataset_name (str): Name of the dataset
            config (dict, optional): Configuration used to generate this dataset
        Returns:
            tuple: (dataset_dir, images_dir) paths
        """
        # Add timestamp to prevent conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir = self.base_dir / f"{dataset_name}_{timestamp}"
        images_dir = dataset_dir / "images"
        
        dataset_dir.mkdir(exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        
        # Save config if provided (for reproducibility)
        if config:
            config_file = dataset_dir / "config.yaml"
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False)
        
        return str(dataset_dir), str(images_dir)

    def create_metadata_file(self, dataset_dir):
        """
        Create a new metadata.jsonl file.
        Args:
            dataset_dir (str): Dataset directory path
        Returns:
            str: Path to metadata file
        """
        metadata_path = Path(dataset_dir) / "metadata.jsonl"
        if not metadata_path.exists():
            metadata_path.touch()
        return str(metadata_path)

    def archive_dataset(self, dataset_name):
        """
        Move a dataset to the archive directory.
        Args:
            dataset_name (str): Name of the dataset to archive
        """
        dataset_dir = next(self.base_dir.glob(f"{dataset_name}_*"), None)
        if dataset_dir and dataset_dir.is_dir():
            archive_path = self.archive_dir / dataset_dir.name
            shutil.move(str(dataset_dir), str(archive_path))
            self.logger.info(f"Archived dataset: {dataset_name} to {archive_path}")
        else:
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")

    def cleanup_tmp(self):
        """Remove temporary files and directories."""
        if self.tmp_dir.exists():
            shutil.rmtree(str(self.tmp_dir))
            self.tmp_dir.mkdir()
            self.logger.info("Cleaned up temporary files")

    def export_dataset(self, dataset_name, output_path=None):
        """
        Export a dataset as a zip file.
        Args:
            dataset_name (str): Name of the dataset to export
            output_path (str, optional): Where to save the zip file
        Returns:
            str: Path to the exported zip file
        """
        dataset_dir = next(self.base_dir.glob(f"{dataset_name}_*"), None)
        if not dataset_dir:
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")
        
        if not output_path:
            output_path = str(self.base_dir / f"{dataset_name}.zip")
        
        shutil.make_archive(
            output_path.replace('.zip', ''),  # Remove .zip as make_archive adds it
            'zip',
            str(dataset_dir)
        )
        
        self.logger.info(f"Exported dataset to: {output_path}")
        return output_path

    def import_dataset(self, zip_path, new_name=None):
        """
        Import a dataset from a zip file.
        Args:
            zip_path (str): Path to the zip file
            new_name (str, optional): New name for the imported dataset
        Returns:
            str: Path to the imported dataset directory
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
        
        # Create temporary extraction directory
        extract_dir = self.tmp_dir / "import_temp"
        extract_dir.mkdir(exist_ok=True)
        
        try:
            # Extract zip
            shutil.unpack_archive(str(zip_path), str(extract_dir))
            
            # Determine target directory name
            if new_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_dir = self.base_dir / f"{new_name}_{timestamp}"
            else:
                # Use original name but add new timestamp
                original_name = next(extract_dir.iterdir()).name.split('_')[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_dir = self.base_dir / f"{original_name}_{timestamp}"
            
            # Move extracted contents to target directory
            extracted_contents = next(extract_dir.iterdir())  # Get the extracted folder
            shutil.move(str(extracted_contents), str(target_dir))
            
            self.logger.info(f"Imported dataset to: {target_dir}")
            return str(target_dir)
            
        finally:
            # Cleanup
            if extract_dir.exists():
                shutil.rmtree(str(extract_dir))

    def list_datasets(self, include_archived=False):
        """
        List all available datasets.
        Args:
            include_archived (bool): Whether to include archived datasets
        Returns:
            dict: Dictionary of dataset information
        """
        datasets = {}
        
        # List active datasets
        for dataset_dir in self.base_dir.glob("*_*/"):
            if dataset_dir.is_dir() and dataset_dir not in [self.archive_dir, self.tmp_dir]:
                name = dataset_dir.name
                metadata_file = dataset_dir / "metadata.jsonl"
                config_file = dataset_dir / "config.yaml"
                
                datasets[name] = {
                    "path": str(dataset_dir),
                    "status": "active",
                    "has_metadata": metadata_file.exists(),
                    "has_config": config_file.exists(),
                    "image_count": len(list((dataset_dir / "images").glob("*.png")))
                }
        
        # List archived datasets if requested
        if include_archived:
            for dataset_dir in self.archive_dir.glob("*_*/"):
                if dataset_dir.is_dir():
                    name = dataset_dir.name
                    metadata_file = dataset_dir / "metadata.jsonl"
                    config_file = dataset_dir / "config.yaml"
                    
                    datasets[name] = {
                        "path": str(dataset_dir),
                        "status": "archived",
                        "has_metadata": metadata_file.exists(),
                        "has_config": config_file.exists(),
                        "image_count": len(list((dataset_dir / "images").glob("*.png")))
                    }
        
        return datasets

    def get_dataset_info(self, dataset_name):
        """
        Get detailed information about a specific dataset.
        Args:
            dataset_name (str): Name of the dataset
        Returns:
            dict: Dataset information
        """
        # Look in active datasets first, then archived
        dataset_dir = next(self.base_dir.glob(f"{dataset_name}_*"), None)
        status = "active"
        
        if not dataset_dir:
            dataset_dir = next(self.archive_dir.glob(f"{dataset_name}_*"), None)
            status = "archived"
        
        if not dataset_dir:
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")
        
        metadata_file = dataset_dir / "metadata.jsonl"
        config_file = dataset_dir / "config.yaml"
        images_dir = dataset_dir / "images"
        
        info = {
            "name": dataset_name,
            "path": str(dataset_dir),
            "status": status,
            "created": datetime.fromtimestamp(dataset_dir.stat().st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
            "modified": datetime.fromtimestamp(dataset_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "has_metadata": metadata_file.exists(),
            "has_config": config_file.exists(),
            "image_count": len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0,
            "total_size": sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file()),
            "config": None,
            "metadata_sample": None
        }
        
        # Load config if available
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                info["config"] = yaml.safe_load(f)
        
        # Load sample of metadata if available
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                first_lines = [next(f) for _ in range(3)]
                info["metadata_sample"] = [json.loads(line) for line in first_lines if line.strip()]
        
        return info

    def delete_dataset(self, dataset_name, archive_first=True):
        """
        Delete a dataset.
        Args:
            dataset_name (str): Name of the dataset to delete
            archive_first (bool): Whether to archive before deletion
        """
        if archive_first:
            try:
                self.archive_dataset(dataset_name)
                dataset_dir = next(self.archive_dir.glob(f"{dataset_name}_*"), None)
            except FileNotFoundError:
                dataset_dir = next(self.base_dir.glob(f"{dataset_name}_*"), None)
        else:
            dataset_dir = next(self.base_dir.glob(f"{dataset_name}_*"), None)
        
        if dataset_dir and dataset_dir.is_dir():
            shutil.rmtree(str(dataset_dir))
            self.logger.info(f"Deleted dataset: {dataset_name}")
        else:
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")
