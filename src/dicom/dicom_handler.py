import os
import pydicom
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut

class DicomHandler:
    """
    A handler class for DICOM file operations, optimized for X-Ray images.
    """
    
    def __init__(self):
        self.dataset = None
        self.image = None
        
    def load_dicom(self, file_path: str) -> np.ndarray:
        """
        Load a DICOM file and return its pixel array.
        
        Args:
            file_path (str): Path to the DICOM file
            
        Returns:
            np.ndarray: The pixel array of the DICOM image
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"DICOM file not found: {file_path}")
            
        self.dataset = pydicom.dcmread(file_path)
        
        # Convert to float for better precision in processing
        self.image = self.dataset.pixel_array.astype(float)
        
        # Apply VOI LUT transformation for proper visualization
        if hasattr(self.dataset, 'WindowCenter') and hasattr(self.dataset, 'WindowWidth'):
            self.image = apply_voi_lut(self.image, self.dataset)
            
        # Normalize to 0-1 range
        if self.image.max() != self.image.min():
            self.image = (self.image - self.image.min()) / (self.image.max() - self.image.min())
            
        return self.image
        
    def save_dicom(self, output_path: str, image: np.ndarray = None) -> None:
        """
        Save the processed image as a DICOM file.
        
        Args:
            output_path (str): Path where to save the DICOM file
            image (np.ndarray, optional): Image array to save. If None, uses the current image
        """
        if image is not None:
            self.image = image
            
        if self.image is None:
            raise ValueError("No image data to save")
            
        if self.dataset is None:
            raise ValueError("No DICOM dataset available")
            
        # Ensure the image data is in the correct format for DICOM
        processed_image = (self.image * 65535).astype(np.uint16)  # Convert to 16-bit
        self.dataset.PixelData = processed_image.tobytes()
        self.dataset.save_as(output_path)
        
    def get_metadata(self) -> dict:
        """
        Return important DICOM metadata.
        
        Returns:
            dict: Dictionary containing relevant DICOM metadata
        """
        if self.dataset is None:
            raise ValueError("No DICOM dataset loaded")
            
        metadata = {
            'PatientID': getattr(self.dataset, 'PatientID', 'Unknown'),
            'StudyDate': getattr(self.dataset, 'StudyDate', 'Unknown'),
            'Modality': getattr(self.dataset, 'Modality', 'Unknown'),
            'Manufacturer': getattr(self.dataset, 'Manufacturer', 'Unknown'),
            'ImageSize': self.image.shape if self.image is not None else None,
        }
        
        return metadata 