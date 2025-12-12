# Cloud Deployment Guide for Kidney Detection App

This guide explains how to properly deploy the Kidney Abnormality Detection application to cloud platforms (Streamlit Cloud, Heroku, AWS, etc.) with the correct OpenCV configuration.

## The OpenCV Issue

The application uses **Ultralytics YOLO** models, which depend on **OpenCV**. By default, `opencv-python` includes GUI dependencies (GTK, Qt) that require display servers and OpenGL libraries (`libGL.so.1`). These are typically not available in cloud deployment environments, causing errors like:

```
ImportError: libGL.so.1: cannot open shared object file
```

## Solution: opencv-python-headless

We use `opencv-python-headless` instead, which provides all OpenCV functionality without GUI dependencies, making it perfect for cloud deployment.

## Deployment Steps

### Option 1: Automatic Setup (Recommended)

Run the setup script that handles everything automatically:

```bash
chmod +x setup_cloud_env.sh
./setup_cloud_env.sh
```

This script will:
1. Remove any existing `opencv-python` (full version)
2. Install `opencv-python-headless` first
3. Install all other dependencies
4. Verify the installation

### Option 2: Manual Setup

If you prefer manual installation:

```bash
# 1. Remove any existing opencv-python packages
pip uninstall -y opencv-python opencv-contrib-python

# 2. Install opencv-python-headless FIRST (before ultralytics)
pip install opencv-python-headless>=4.8.0

# 3. Install remaining dependencies
pip install -r requirements.txt
```

### Verification

After installation, verify your setup:

```bash
python verify_opencv.py
```

This script checks:
- ✅ OpenCV version and build info
- ✅ Whether GUI dependencies are present (should be NO)
- ✅ Which OpenCV packages are installed
- ✅ Ultralytics compatibility
- ✅ YOLO import functionality

If you deploy on Streamlit Cloud, also ensure `packages.txt` is present so `libgl1` and `libglib2.0-0` are installed.

Expected output for successful deployment:
```
✅ DEPLOYMENT CHECK PASSED
   opencv-python-headless is correctly installed
   Your application is ready for cloud deployment!
```

## Platform-Specific Instructions

### Streamlit Cloud

1. Ensure `requirements.txt` lists `opencv-python-headless` **before** `ultralytics`
2. Push to GitHub
3. Deploy via Streamlit Cloud dashboard
4. The platform will automatically use the requirements.txt

### Heroku

Add to `Procfile`:
```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

Add `runtime.txt`:
```
python-3.11
```

### Docker

Example `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (opencv-python-headless will be used)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### AWS EC2 / Google Cloud / Azure

Use the setup script or manual installation steps above, then:

```bash
# Start the app
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

## Troubleshooting

### Issue: "libGL.so.1: cannot open shared object file"

**Cause:** `opencv-python` (full version) is installed instead of `opencv-python-headless`

**Fix:**
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python-headless>=4.8.0
pip install --force-reinstall ultralytics
```

### Issue: Models not loading

**Cause:** Model files might not be in the correct location

**Fix:**
- Ensure `models/coronalBest weights.pt` exists
- Ensure `models/axialBest Weights.pt` exists
- Check file permissions

### Issue: Application shows "Using placeholder detection"

**Possible causes:**
1. Model files are missing
2. Ultralytics failed to import
3. OpenCV configuration issue

**Debugging:**
```bash
python verify_opencv.py
```

Check the app sidebar for ultralytics status.

## Requirements File Explanation

```txt
# Core dependencies
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0

# CRITICAL: OpenCV headless for cloud deployment
# This MUST be installed before ultralytics
opencv-python-headless>=4.8.0

# YOLO model dependencies
# ultralytics may try to install opencv-python, but
# opencv-python-headless should satisfy the requirement
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
```

The order matters! `opencv-python-headless` must come **before** `ultralytics` so pip doesn't try to install the full `opencv-python` package.

## Application Features

The app gracefully handles OpenCV issues:

1. **Early Detection:** Checks ultralytics availability at startup
2. **Fallback Mode:** Uses placeholder detections if YOLO models can't load
3. **User Feedback:** Shows clear messages about the deployment status
4. **Graceful Degradation:** App remains functional even without real YOLO inference

## Production Checklist

Before deploying to production:

- [ ] Run `python verify_opencv.py` - should pass all checks
- [ ] Test model loading locally
- [ ] Verify both coronal and axial models exist
- [ ] Test image upload and processing
- [ ] Check app performance with sample images
- [ ] Review error handling and user messages
- [ ] Ensure model files are included in deployment
- [ ] Set appropriate memory limits for your platform

## Support

If you encounter issues not covered here:

1. Check the logs for specific error messages
2. Run `python verify_opencv.py` for diagnostic information
3. Verify all model files are present and accessible
4. Ensure sufficient memory is allocated (YOLO models require ~2GB RAM)

## Additional Resources

- [OpenCV Headless Documentation](https://pypi.org/project/opencv-python-headless/)
- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)
