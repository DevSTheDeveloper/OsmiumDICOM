import cv2
import numpy as np
from typing import Tuple, Optional

class XRayProcessor:
    """
    A processor class for X-Ray image enhancement and analysis.
    Optimized for Apple Silicon using OpenCV's hardware acceleration.
    """
    
    def __init__(self):
        # Enable OpenCV hardware acceleration if available
        cv2.setUseOptimized(True)
        
    def enhance(self, image: np.ndarray, 
                contrast: float = 1.2,
                brightness: float = 0.0,
                denoise_strength: float = 10.0) -> np.ndarray:
        """
        Enhance X-Ray image quality with multiple techniques.
        
        Args:
            image (np.ndarray): Input image array
            contrast (float): Contrast enhancement factor
            brightness (float): Brightness adjustment value
            denoise_strength (float): Strength of denoising
            
        Returns:
            np.ndarray: Enhanced image
        """
        # Ensure image is in float format
        if image.dtype != np.float32:
            image = image.astype(np.float32)
            
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(
            (image * 255).astype(np.uint8),
            None,
            h=denoise_strength,
            searchWindowSize=21,
            templateWindowSize=7
        ).astype(np.float32) / 255.0
        
        # Apply contrast and brightness adjustments
        enhanced = cv2.convertScaleAbs(
            denoised,
            alpha=contrast,
            beta=brightness
        ).astype(np.float32) / 255.0
        
        # Apply adaptive histogram equalization
        enhanced = self._apply_clahe(enhanced)
        
        return enhanced
        
    def _apply_clahe(self, image: np.ndarray,
                     clip_limit: float = 2.0,
                     grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Args:
            image (np.ndarray): Input image
            clip_limit (float): Threshold for contrast limiting
            grid_size (tuple): Size of grid for histogram equalization
            
        Returns:
            np.ndarray: CLAHE enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply((image * 255).astype(np.uint8)).astype(np.float32) / 255.0
        
    def detect_anomalies(self, image: np.ndarray,
                        threshold: float = 0.2) -> Tuple[np.ndarray, list]:
        """
        Detect potential anomalies in X-Ray images.
        
        Args:
            image (np.ndarray): Input image
            threshold (float): Detection threshold
            
        Returns:
            Tuple[np.ndarray, list]: Marked image and list of detected anomaly regions
        """
        # Convert to uint8 for OpenCV operations
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            img_uint8,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours based on area
        min_area = image.shape[0] * image.shape[1] * threshold
        significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Draw contours on original image
        result = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(result, significant_contours, -1, (0, 255, 0), 2)
        
        # Extract regions of interest
        regions = []
        for contour in significant_contours:
            x, y, w, h = cv2.boundingRect(contour)
            regions.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': cv2.contourArea(contour)
            })
            
        return result.astype(np.float32) / 255.0, regions 