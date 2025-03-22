import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Union
import numpy as np

class DatasetManager:
    """
    Manages dataset organization, tagging, and preparation for model training.
    """
    
    def __init__(self, base_path: str):
        """
        Initialize dataset manager.
        
        Args:
            base_path (str): Base directory for storing datasets
        """
        self.base_path = base_path
        self.dataset_path = os.path.join(base_path, 'datasets')
        self.annotations_path = os.path.join(base_path, 'annotations')
        
        # Create necessary directories
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.annotations_path, exist_ok=True)
        
        # Load existing annotations
        self.annotations = self._load_annotations()
        
    def add_image(self, image_path: str, dataset_name: str, copy: bool = True) -> str:
        """
        Add an image to the dataset.
        
        Args:
            image_path (str): Path to the DICOM image
            dataset_name (str): Name of the dataset
            copy (bool): Whether to copy the file or move it
            
        Returns:
            str: ID of the added image
        """
        dataset_dir = os.path.join(self.dataset_path, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Generate unique ID for the image
        image_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Copy or move the image
        dest_path = os.path.join(dataset_dir, f"{image_id}.dcm")
        if copy:
            shutil.copy2(image_path, dest_path)
        else:
            shutil.move(image_path, dest_path)
            
        # Initialize annotation entry
        self.annotations[image_id] = {
            'dataset': dataset_name,
            'path': dest_path,
            'tags': [],
            'regions': [],
            'metadata': {}
        }
        
        self._save_annotations()
        return image_id
        
    def add_tags(self, image_id: str, tags: List[str]) -> None:
        """
        Add tags to an image.
        
        Args:
            image_id (str): ID of the image
            tags (List[str]): List of tags to add
        """
        if image_id not in self.annotations:
            raise ValueError(f"Image ID {image_id} not found")
            
        self.annotations[image_id]['tags'].extend(tags)
        self.annotations[image_id]['tags'] = list(set(self.annotations[image_id]['tags']))
        self._save_annotations()
        
    def add_region(self, image_id: str, region: Dict[str, Union[int, float]], 
                  label: str, confidence: float = 1.0) -> None:
        """
        Add an annotated region to an image.
        
        Args:
            image_id (str): ID of the image
            region (dict): Region coordinates (x, y, width, height)
            label (str): Label for the region
            confidence (float): Confidence score for the annotation
        """
        if image_id not in self.annotations:
            raise ValueError(f"Image ID {image_id} not found")
            
        region_entry = {
            'region': region,
            'label': label,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        self.annotations[image_id]['regions'].append(region_entry)
        self._save_annotations()
        
    def get_dataset_statistics(self, dataset_name: Optional[str] = None) -> Dict:
        """
        Get statistics about the dataset.
        
        Args:
            dataset_name (str, optional): Specific dataset to analyze
            
        Returns:
            dict: Dataset statistics
        """
        stats = {
            'total_images': 0,
            'total_tags': set(),
            'tags_frequency': {},
            'regions_per_image': [],
            'datasets': set()
        }
        
        for img_id, anno in self.annotations.items():
            if dataset_name and anno['dataset'] != dataset_name:
                continue
                
            stats['total_images'] += 1
            stats['datasets'].add(anno['dataset'])
            stats['regions_per_image'].append(len(anno['regions']))
            
            for tag in anno['tags']:
                stats['total_tags'].add(tag)
                stats['tags_frequency'][tag] = stats['tags_frequency'].get(tag, 0) + 1
                
        stats['total_tags'] = list(stats['total_tags'])
        stats['datasets'] = list(stats['datasets'])
        stats['avg_regions_per_image'] = np.mean(stats['regions_per_image'])
        
        return stats
        
    def prepare_training_data(self, dataset_name: str) -> Dict:
        """
        Prepare data for model training.
        
        Args:
            dataset_name (str): Name of the dataset to prepare
            
        Returns:
            dict: Training data information
        """
        training_data = {
            'images': [],
            'annotations': [],
            'classes': set()
        }
        
        for img_id, anno in self.annotations.items():
            if anno['dataset'] != dataset_name:
                continue
                
            training_data['images'].append({
                'id': img_id,
                'path': anno['path'],
                'tags': anno['tags']
            })
            
            for region in anno['regions']:
                training_data['annotations'].append({
                    'image_id': img_id,
                    'region': region['region'],
                    'label': region['label'],
                    'confidence': region['confidence']
                })
                training_data['classes'].add(region['label'])
                
        training_data['classes'] = list(training_data['classes'])
        return training_data
        
    def _load_annotations(self) -> Dict:
        """Load annotations from disk."""
        annotation_file = os.path.join(self.annotations_path, 'annotations.json')
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_annotations(self) -> None:
        """Save annotations to disk."""
        annotation_file = os.path.join(self.annotations_path, 'annotations.json')
        with open(annotation_file, 'w') as f:
            json.dump(self.annotations, f, indent=2) 