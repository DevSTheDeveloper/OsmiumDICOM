import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                           QListWidget, QFileDialog, QMessageBox, QScrollArea,
                           QFrame, QSplitter, QInputDialog, QComboBox)
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen, QImage
import numpy as np
import matplotlib.pyplot as plt

from ..utils.dataset_manager import DatasetManager
from ..utils.image_handler import ImageHandler
from ..processing.xray_processor import XRayProcessor

class RegionSelector(QLabel):
    """Widget for selecting regions in the image."""
    
    def __init__(self):
        super().__init__()
        self.begin = None
        self.end = None
        self.is_drawing = False
        self.regions = []
        self.current_label = ""
        
    def set_label(self, label: str):
        self.current_label = label
        
    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = self.begin
        self.is_drawing = True
        
    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.end = event.pos()
            self.update()
            
    def mouseReleaseEvent(self, event):
        if self.is_drawing:
            self.is_drawing = False
            if self.current_label and self.begin and self.end:
                region = {
                    'x': min(self.begin.x(), self.end.x()),
                    'y': min(self.begin.y(), self.end.y()),
                    'width': abs(self.begin.x() - self.end.x()),
                    'height': abs(self.begin.y() - self.end.y())
                }
                self.regions.append((region, self.current_label))
                self.update()
                
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        
        for region, label in self.regions:
            painter.drawRect(
                region['x'], region['y'],
                region['width'], region['height']
            )
            painter.drawText(
                region['x'], region['y'] - 5,
                label
            )
            
        if self.is_drawing and self.begin and self.end:
            painter.drawRect(QRect(self.begin, self.end))
            
    def clear_regions(self):
        self.regions = []
        self.update()

class TaggerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_handler = ImageHandler()
        self.xray_processor = XRayProcessor()
        self.dataset_manager = None
        self.current_image_id = None
        self.current_image_path = None
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Medical Image Tagger')
        self.setGeometry(100, 100, 1200, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Dataset controls
        dataset_frame = QFrame()
        dataset_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        dataset_layout = QVBoxLayout(dataset_frame)
        
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setPlaceholderText("Dataset path...")
        dataset_layout.addWidget(self.dataset_path_edit)
        
        dataset_btn_layout = QHBoxLayout()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_dataset)
        load_btn = QPushButton("Load Dataset")
        load_btn.clicked.connect(self.load_dataset)
        dataset_btn_layout.addWidget(browse_btn)
        dataset_btn_layout.addWidget(load_btn)
        dataset_layout.addLayout(dataset_btn_layout)
        
        left_layout.addWidget(dataset_frame)
        
        # Image controls
        image_frame = QFrame()
        image_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        image_layout = QVBoxLayout(image_frame)
        
        # File type selector
        file_type_layout = QHBoxLayout()
        file_type_label = QLabel("File Type:")
        self.file_type_combo = QComboBox()
        self.file_type_combo.addItems([
            "DICOM Files (*.dcm)",
            "PNG Files (*.png)",
            "All Files (*)"
        ])
        file_type_layout.addWidget(file_type_label)
        file_type_layout.addWidget(self.file_type_combo)
        image_layout.addLayout(file_type_layout)
        
        load_image_btn = QPushButton("Load Image")
        load_image_btn.clicked.connect(self.load_image)
        image_layout.addWidget(load_image_btn)
        
        # Export controls
        export_layout = QHBoxLayout()
        export_label = QLabel("Export as:")
        self.export_combo = QComboBox()
        self.export_combo.addItems(["DICOM", "PNG"])
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.export_image)
        export_layout.addWidget(export_label)
        export_layout.addWidget(self.export_combo)
        export_layout.addWidget(export_btn)
        image_layout.addLayout(export_layout)
        
        # Tags section
        tags_label = QLabel("Tags:")
        image_layout.addWidget(tags_label)
        
        self.tags_list = QListWidget()
        image_layout.addWidget(self.tags_list)
        
        tag_btn_layout = QHBoxLayout()
        add_tag_btn = QPushButton("Add Tag")
        add_tag_btn.clicked.connect(self.add_tag)
        remove_tag_btn = QPushButton("Remove Tag")
        remove_tag_btn.clicked.connect(self.remove_tag)
        tag_btn_layout.addWidget(add_tag_btn)
        tag_btn_layout.addWidget(remove_tag_btn)
        image_layout.addLayout(tag_btn_layout)
        
        left_layout.addWidget(image_frame)
        
        # Region controls
        region_frame = QFrame()
        region_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        region_layout = QVBoxLayout(region_frame)
        
        add_region_btn = QPushButton("Add Region")
        add_region_btn.clicked.connect(self.start_region_selection)
        clear_regions_btn = QPushButton("Clear Regions")
        clear_regions_btn.clicked.connect(self.clear_regions)
        region_layout.addWidget(add_region_btn)
        region_layout.addWidget(clear_regions_btn)
        
        left_layout.addWidget(region_frame)
        
        # Save button
        save_btn = QPushButton("Save Changes")
        save_btn.clicked.connect(self.save_changes)
        left_layout.addWidget(save_btn)
        
        splitter.addWidget(left_panel)
        
        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.image_label = RegionSelector()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll_area.setWidget(self.image_label)
        right_layout.addWidget(scroll_area)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])
        
    def browse_dataset(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Dataset Directory"
        )
        if directory:
            self.dataset_path_edit.setText(directory)
            
    def load_dataset(self):
        path = self.dataset_path_edit.text()
        if not path:
            QMessageBox.warning(self, "Error", "Please specify a dataset path")
            return
            
        self.dataset_manager = DatasetManager(path)
        QMessageBox.information(
            self, "Success", "Dataset loaded successfully"
        )
        
    def load_image(self):
        if not self.dataset_manager:
            QMessageBox.warning(
                self, "Error", "Please load a dataset first"
            )
            return
            
        file_filter = self.file_type_combo.currentText()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", file_filter
        )
        
        if file_path:
            try:
                dataset_name, ok = QInputDialog.getText(
                    self, "Dataset Name",
                    "Enter dataset name for this image:"
                )
                if not ok or not dataset_name:
                    return
                    
                self.current_image_id = self.dataset_manager.add_image(
                    file_path, dataset_name
                )
                self.current_image_path = file_path
                
                image = self.image_handler.load_image(file_path)
                enhanced = self.xray_processor.enhance(image)
                
                plt.imsave('temp.png', enhanced, cmap='gray')
                pixmap = QPixmap('temp.png')
                self.image_label.setPixmap(pixmap)
                os.remove('temp.png')
                
                self.tags_list.clear()
                self.image_label.clear_regions()
                
                metadata = self.image_handler.get_metadata(file_path)
                self.show_metadata(metadata)
                
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to load image: {str(e)}"
                )
                
    def export_image(self):
        if not self.current_image_path:
            QMessageBox.warning(
                self, "Error",
                "No image loaded"
            )
            return
            
        export_format = self.export_combo.currentText()
        ext = ".dcm" if export_format == "DICOM" else ".png"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "",
            f"{export_format} Files (*{ext})"
        )
        
        if file_path:
            try:
                image = self.image_handler.load_image(self.current_image_path)
                self.image_handler.save_image(
                    image, file_path,
                    original_path=self.current_image_path
                )
                QMessageBox.information(
                    self, "Success",
                    "Image exported successfully"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to export image: {str(e)}"
                )
                
    def show_metadata(self, metadata: dict):
        metadata_str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
        QMessageBox.information(
            self, "Image Metadata",
            metadata_str
        )
                
    def add_tag(self):
        if not self.current_image_id:
            QMessageBox.warning(
                self, "Error",
                "Please load an image first"
            )
            return
            
        tag, ok = QInputDialog.getText(
            self, "Add Tag",
            "Enter tag name:"
        )
        
        if ok and tag:
            self.dataset_manager.add_tags(
                self.current_image_id,
                [tag]
            )
            self.tags_list.addItem(tag)
            
    def remove_tag(self):
        current_item = self.tags_list.currentItem()
        if current_item:
            self.tags_list.takeItem(self.tags_list.row(current_item))
            
    def start_region_selection(self):
        label, ok = QInputDialog.getText(
            self, "Region Label",
            "Enter label for the region:"
        )
        
        if ok and label:
            self.image_label.set_label(label)
            
    def clear_regions(self):
        self.image_label.clear_regions()
        
    def save_changes(self):
        if not self.current_image_id:
            QMessageBox.warning(
                self, "Error",
                "No image loaded"
            )
            return
            
        try:
            for region, label in self.image_label.regions:
                self.dataset_manager.add_region(
                    self.current_image_id,
                    region,
                    label
                )
                
            QMessageBox.information(
                self, "Success",
                "Changes saved successfully"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to save changes: {str(e)}"
            )

def main():
    app = QApplication(sys.argv)
    tagger = TaggerApp()
    tagger.show()
    sys.exit(app.exec()) 