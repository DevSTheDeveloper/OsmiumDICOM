# OsmiumDICOM

A powerful DICOM processing toolkit optimized for Apple Silicon Macs, designed for X-Ray image processing and analysis.

## Features

- DICOM file reading and writing
- X-Ray image processing and enhancement
- Local processing optimization for Apple Silicon
- Support for common DICOM operations

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
OsmiumDICOM/
├── src/
│   ├── dicom/          # DICOM handling utilities
│   ├── processing/     # Image processing modules
│   └── utils/          # Helper functions
├── tests/              # Unit tests
├── examples/           # Usage examples
└── requirements.txt    # Project dependencies
```

## Usage

Basic example of loading and processing a DICOM file:

```python
from src.dicom import DicomHandler
from src.processing import XRayProcessor

# Load DICOM file
handler = DicomHandler()
image_data = handler.load_dicom("path/to/image.dcm")

# Process X-Ray image
processor = XRayProcessor()
enhanced_image = processor.enhance(image_data)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.