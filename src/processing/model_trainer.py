import os
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pydicom
from ..utils.dataset_manager import DatasetManager

class XRayDataset(Dataset):
    """Custom Dataset for X-Ray images with annotations."""
    
    def __init__(self, data_info: Dict, transform=None):
        self.data = data_info['images']
        self.annotations = data_info['annotations']
        self.classes = data_info['classes']
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img_info = self.data[idx]
        
        # Load DICOM image
        ds = pydicom.dcmread(img_info['path'])
        image = ds.pixel_array.astype(float)
        
        # Normalize image
        if image.max() != image.min():
            image = (image - image.min()) / (image.max() - image.min())
        
        # Convert to RGB (3 channels)
        image = np.stack([image] * 3, axis=0)
        
        if self.transform:
            image = self.transform(torch.FloatTensor(image))
            
        # Create label tensor
        labels = torch.zeros(len(self.classes))
        for tag in img_info['tags']:
            if tag in self.class_to_idx:
                labels[self.class_to_idx[tag]] = 1
                
        return image, labels

class ModelTrainer:
    """Handles model training and evaluation for X-Ray classification."""
    
    def __init__(self, dataset_manager: DatasetManager, model_dir: str):
        """
        Initialize the model trainer.
        
        Args:
            dataset_manager: DatasetManager instance
            model_dir: Directory to save trained models
        """
        self.dataset_manager = dataset_manager
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Set device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    def train_model(self, dataset_name: str, 
                   num_epochs: int = 10,
                   batch_size: int = 8,
                   learning_rate: float = 0.001,
                   model_name: Optional[str] = None) -> Dict:
        """
        Train a model on the specified dataset.
        
        Args:
            dataset_name: Name of the dataset to train on
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            model_name: Name for saving the model
            
        Returns:
            dict: Training history
        """
        # Prepare data
        data_info = self.dataset_manager.prepare_training_data(dataset_name)
        if not data_info['images']:
            raise ValueError("No training data available")
            
        # Create dataset and dataloader
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        dataset = XRayDataset(data_info, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=2)
        
        # Initialize model
        model = models.resnet50(pretrained=True)
        num_classes = len(data_info['classes'])
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        history = {
            'loss': [],
            'accuracy': []
        }
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Calculate accuracy
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.numel()
                
            # Record metrics
            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct / total
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Loss: {avg_loss:.4f} "
                  f"Accuracy: {accuracy:.4f}")
        
        # Save model
        if model_name is None:
            model_name = f"xray_model_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        model_path = os.path.join(self.model_dir, f"{model_name}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'classes': data_info['classes'],
            'history': history,
            'model_config': {
                'num_classes': num_classes,
                'architecture': 'resnet50'
            }
        }, model_path)
        
        return history
        
    def predict(self, model_name: str, image: np.ndarray) -> Dict[str, float]:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the model to use
            image: Input image array
            
        Returns:
            dict: Class probabilities
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.pth")
        if not os.path.exists(model_path):
            raise ValueError(f"Model not found: {model_name}")
            
        # Load model
        checkpoint = torch.load(model_path)
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(checkpoint['classes']))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Prepare image
        if image.max() != image.min():
            image = (image - image.min()) / (image.max() - image.min())
        image = np.stack([image] * 3, axis=0)
        
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(torch.FloatTensor(image)).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.sigmoid(outputs)[0]
            
        # Return results
        return {
            cls: prob.item()
            for cls, prob in zip(checkpoint['classes'], probabilities)
        } 