# Kidney Abnormality Detection

A deep learning-based web application that uses YOLO object detection models to identify and localize kidney abnormalities (cysts, stones, and tumors) in CT scan images.

---

## Features

- **Multi-view Detection**: Dedicated YOLO models for Coronal and Axial CT scan views
- **Four Detection Classes**: Kidney, Cyst, Stone, and Tumor
- **Single & Batch Processing**: Analyze one image or multiple images at once
- **Adjustable Confidence Threshold**: Filter detections by confidence score
- **ROI Selection**: Define a region of interest for focused analysis
- **Segmentation Mask Support**: Visualize segmentation masks when available
- **Cloud-Ready**: Optimized for headless deployment environments
- **Demo Mode**: Falls back to placeholder detections when models are unavailable

---

## Application Pages

| Page | Description |
|------|-------------|
| **Detection** | Upload and analyze CT scan images (single or batch mode) |
| **About** | Project background and clinical context |
| **Dataset** | Dataset statistics, class distributions, and augmentation settings |
| **Model** | Model architecture and training configuration details |

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- pip

### Installation

**Option 1 – Automatic Setup (recommended)**

```bash
chmod +x setup_cloud_env.sh
./setup_cloud_env.sh
```

**Option 2 – Manual Setup**

```bash
# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install the correct OpenCV build first
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-python-headless==4.11.0.86

# Install remaining dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python verify_opencv.py
```

### Run the Application

```bash
streamlit run streamlit_app.py
```

The app will be available at **http://localhost:8501**.

---

## Usage

1. Open the app in your browser.
2. Go to the **Detection** page.
3. Select the CT scan view type: **Coronal** or **Axial**.
4. Upload one or more CT scan images (PNG, JPG, JPEG, TIFF, BMP — max 10 MB each).
5. Adjust the confidence threshold slider as needed.
6. View detection results with bounding boxes, class labels, and confidence scores.

---

## Dataset

| Split | Images | Percentage |
|-------|--------|------------|
| Training | 12,924 | 70% |
| Validation | 612 | 10% |
| Test | 1,225 | 20% |
| **Total** | **14,761** | |

**View Distribution**

| View | Images | Share |
|------|--------|-------|
| Axial | 8,514 | 57.7% |
| Coronal | 6,247 | 42.3% |

**Abnormality Distribution (training set)**

| Class | Training | Validation | Test |
|-------|----------|------------|------|
| Cysts | 3,636 | 170 | 341 |
| Stones | 4,500 | 214 | 428 |
| Tumors | 4,788 | 228 | 456 |

**Preprocessing & Augmentation**

- Images resized to **640 × 640** pixels
- 3× augmentation per training image:
  - Brightness variation: ±25%
  - Gaussian blur: up to 0.8 px
  - Salt-and-pepper noise: up to 0.54% of pixels

---

## Models

Two YOLOv8 segmentation models are included in the `models/` directory:

| File | View | Size |
|------|------|------|
| `coronalBest weights.pt` | Coronal | ~5.8 MB |
| `axialBest Weights.pt` | Axial | ~5.8 MB |

Both models detect four classes: **kidney**, **cyst**, **stone**, **tumor**.

---

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for full instructions covering:

- Streamlit Cloud
- Heroku
- Docker
- AWS / Google Cloud / Azure

**Quick Docker example**

```bash
docker build -t kidney-detection .
docker run -p 8501:8501 kidney-detection
```

> **Important:** Always use `opencv-python-headless` instead of `opencv-python` in cloud/container environments to avoid `libGL.so.1` errors.

---

## Project Structure

```
KidneyDetection/
├── models/
│   ├── coronalBest weights.pt   # Coronal YOLO model
│   └── axialBest Weights.pt     # Axial YOLO model
├── streamlit_app.py             # Main Streamlit application
├── datasetvalues.py             # Dataset metadata and configuration
├── verify_opencv.py             # OpenCV installation verification
├── setup_cloud_env.sh           # Automated cloud environment setup
├── requirements.txt             # Python dependencies
├── packages.txt                 # System-level dependencies
└── DEPLOYMENT.md                # Cloud deployment guide
```

---

## Dependencies

Key packages (see `requirements.txt` for pinned versions):

- [Streamlit](https://streamlit.io/) — web application framework
- [Ultralytics](https://docs.ultralytics.com/) — YOLOv8 detection framework
- [PyTorch](https://pytorch.org/) — deep learning backend
- [OpenCV Headless](https://pypi.org/project/opencv-python-headless/) — computer vision
- [Pillow](https://pillow.readthedocs.io/) — image I/O
- [NumPy](https://numpy.org/) / [Pandas](https://pandas.pydata.org/) — data processing
- [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/) — visualization
