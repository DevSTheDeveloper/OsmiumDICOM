import os
import matplotlib.pyplot as plt
from src.dicom.dicom_handler import DicomHandler
from src.processing.xray_processor import XRayProcessor

def process_xray(input_path: str, output_path: str):
    """
    Process an X-Ray DICOM file and save the enhanced version.
    
    Args:
        input_path (str): Path to input DICOM file
        output_path (str): Path to save processed DICOM file
    """
    # Initialize handlers
    dicom_handler = DicomHandler()
    xray_processor = XRayProcessor()
    
    # Load DICOM file
    print(f"Loading DICOM file: {input_path}")
    image = dicom_handler.load_dicom(input_path)
    
    # Get and display metadata
    metadata = dicom_handler.get_metadata()
    print("\nDICOM Metadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")
    
    # Enhance image
    print("\nEnhancing image...")
    enhanced_image = xray_processor.enhance(
        image,
        contrast=1.2,
        brightness=0.1,
        denoise_strength=15.0
    )
    
    # Detect anomalies
    print("Detecting anomalies...")
    marked_image, regions = xray_processor.detect_anomalies(enhanced_image)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(enhanced_image, cmap='gray')
    axes[1].set_title('Enhanced')
    axes[1].axis('off')
    
    axes[2].imshow(marked_image)
    axes[2].set_title('Anomaly Detection')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save results
    print(f"\nSaving processed DICOM to: {output_path}")
    dicom_handler.save_dicom(output_path, enhanced_image)
    
    # Save visualization
    viz_path = os.path.splitext(output_path)[0] + '_visualization.png'
    plt.savefig(viz_path)
    print(f"Saved visualization to: {viz_path}")
    
    # Print detected regions
    if regions:
        print("\nDetected anomaly regions:")
        for i, region in enumerate(regions, 1):
            print(f"Region {i}:")
            print(f"  Position: ({region['x']}, {region['y']})")
            print(f"  Size: {region['width']}x{region['height']}")
            print(f"  Area: {region['area']:.2f} pixels")
    else:
        print("\nNo significant anomalies detected.")

if __name__ == "__main__":
    # Example usage
    input_file = "path/to/your/xray.dcm"
    output_file = "path/to/save/processed_xray.dcm"
    
    process_xray(input_file, output_file) 