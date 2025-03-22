import os
from src.utils.dataset_manager import DatasetManager
from src.processing.model_trainer import ModelTrainer
from src.dicom.dicom_handler import DicomHandler
from src.processing.xray_processor import XRayProcessor

def main():
    # Initialize managers
    base_path = "research_data"
    dataset_manager = DatasetManager(base_path)
    model_trainer = ModelTrainer(dataset_manager, os.path.join(base_path, "models"))
    
    # Example: Add images to dataset
    dataset_name = "chest_xrays"
    image_paths = [
        "path/to/xray1.dcm",
        "path/to/xray2.dcm",
        "path/to/xray3.dcm"
    ]
    
    print("Adding images to dataset...")
    image_ids = []
    for path in image_paths:
        try:
            image_id = dataset_manager.add_image(path, dataset_name)
            image_ids.append(image_id)
            print(f"Added image: {path} with ID: {image_id}")
        except FileNotFoundError:
            print(f"File not found: {path}")
            continue
    
    # Example: Add tags to images
    print("\nAdding tags to images...")
    tags = {
        image_ids[0]: ["pneumonia", "right_lung"],
        image_ids[1]: ["normal"],
        image_ids[2]: ["pneumonia", "left_lung", "infiltration"]
    }
    
    for image_id, image_tags in tags.items():
        dataset_manager.add_tags(image_id, image_tags)
        print(f"Added tags for image {image_id}: {image_tags}")
    
    # Example: Add regions of interest
    print("\nAdding regions of interest...")
    regions = {
        image_ids[0]: [
            {
                "region": {"x": 100, "y": 150, "width": 50, "height": 50},
                "label": "pneumonia",
                "confidence": 0.9
            }
        ]
    }
    
    for image_id, image_regions in regions.items():
        for region_info in image_regions:
            dataset_manager.add_region(
                image_id,
                region_info["region"],
                region_info["label"],
                region_info["confidence"]
            )
            print(f"Added region for image {image_id}: {region_info['label']}")
    
    # Get dataset statistics
    print("\nDataset Statistics:")
    stats = dataset_manager.get_dataset_statistics(dataset_name)
    print(f"Total Images: {stats['total_images']}")
    print(f"Total Tags: {len(stats['total_tags'])}")
    print("Tag Frequencies:")
    for tag, freq in stats['tags_frequency'].items():
        print(f"  {tag}: {freq}")
    
    # Train model
    print("\nTraining model...")
    try:
        history = model_trainer.train_model(
            dataset_name,
            num_epochs=10,
            batch_size=4,
            learning_rate=0.001,
            model_name="chest_xray_classifier"
        )
        
        print("\nTraining completed!")
        print("Final metrics:")
        print(f"Loss: {history['loss'][-1]:.4f}")
        print(f"Accuracy: {history['accuracy'][-1]:.4f}")
        
    except ValueError as e:
        print(f"Training error: {e}")
    
    # Example: Make predictions
    print("\nMaking predictions on a new image...")
    try:
        # Load and process a new image
        dicom_handler = DicomHandler()
        xray_processor = XRayProcessor()
        
        test_image_path = "path/to/test_xray.dcm"
        image = dicom_handler.load_dicom(test_image_path)
        processed_image = xray_processor.enhance(image)
        
        # Get predictions
        predictions = model_trainer.predict("chest_xray_classifier", processed_image)
        
        print("\nPredictions:")
        for class_name, probability in predictions.items():
            print(f"{class_name}: {probability:.2%}")
            
    except FileNotFoundError:
        print(f"Test image not found: {test_image_path}")
    except ValueError as e:
        print(f"Prediction error: {e}")

if __name__ == "__main__":
    main() 