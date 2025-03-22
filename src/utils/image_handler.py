import os
import numpy as np
import pydicom
from PIL import Image
from ..dicom.dicom_handler import DicomHandler
from ..processing.xray_processor import XRayProcessor

class ImageHandler:
    """Unified handler for both DICOM and PNG images."""
    
    def __init__(self):
        self.dicom_handler = DicomHandler()
        self.xray_processor = XRayProcessor()
        
    def load_image(self, file_path: str) -> np.ndarray:
        """
        Load an image file (DICOM or PNG) and return normalized numpy array.
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            np.ndarray: Normalized image array
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
            
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.dcm':
            # Handle DICOM files
            return self.dicom_handler.load_dicom(file_path)
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            # Handle PNG and other image files
            return self._load_png(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
    def save_image(self, image: np.ndarray, output_path: str,
                  original_path: str = None) -> None:
        """
        Save an image array to a file.
        
        Args:
            image (np.ndarray): Image array to save
            output_path (str): Path where to save the image
            original_path (str, optional): Path to original file for metadata
        """
        output_ext = os.path.splitext(output_path)[1].lower()
        
        if output_ext == '.dcm' and original_path and original_path.endswith('.dcm'):
            # Save as DICOM, preserving original metadata
            self.dicom_handler.save_dicom(output_path, image)
        else:
            # Save as PNG
            self._save_png(image, output_path)
            
    def _load_png(self, file_path: str) -> np.ndarray:
        """
        Load a PNG image and convert to normalized numpy array.
        
        Args:
            file_path (str): Path to the PNG file
            
        Returns:
            np.ndarray: Normalized image array
        """
        with Image.open(file_path) as img:
            # Convert to grayscale if needed
            if img.mode != 'L':
                img = img.convert('L')
            
            # Convert to numpy array and normalize
            image = np.array(img).astype(float)
            if image.max() != image.min():
                image = (image - image.min()) / (image.max() - image.min())
                
            return image
            
    def _save_png(self, image: np.ndarray, output_path: str) -> None:
        """
        Save an image array as PNG.
        
        Args:
            image (np.ndarray): Image array to save
            output_path (str): Path where to save the PNG
        """
        # Ensure image is in 0-255 range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
            
        # Save as PNG
        Image.fromarray(image).save(output_path)
        
    def get_metadata(self, file_path: str) -> dict:
        """
        Get metadata from the image file.
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            dict: Image metadata
        """
        if file_path.lower().endswith('.dcm'):
            return self.dicom_handler.get_metadata()
        else:
            # Basic metadata for PNG files
            with Image.open(file_path) as img:
                return {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'filename': os.path.basename(file_path)
                } 