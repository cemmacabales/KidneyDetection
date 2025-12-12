#!/bin/bash
# Cloud Deployment Environment Setup Script
# This script ensures opencv-python-headless is properly installed for cloud deployment

echo "🔧 Setting up cloud deployment environment..."

# Uninstall any existing opencv-python (full version with GUI)
echo "📦 Removing opencv-python if it exists..."
pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true

# Install opencv-python-headless first (before ultralytics)
echo "📦 Installing opencv-python-headless (pinned)..."
pip install --no-cache-dir opencv-python-headless==4.11.0.86

# Install remaining dependencies (pinned in requirements.txt)
echo "📦 Installing remaining dependencies..."
pip install --no-cache-dir -r requirements.txt

# Verify installation
echo ""
echo "✅ Verifying installation..."
python -c "
import cv2
print(f'OpenCV version: {cv2.__version__}')
print(f'OpenCV build info includes GUI: {\"WITH_GTK\" in cv2.getBuildInformation() or \"WITH_QT\" in cv2.getBuildInformation()}')

try:
    import ultralytics
    print(f'Ultralytics version: {ultralytics.__version__}')
    print('✅ All packages installed successfully!')
except Exception as e:
    print(f'⚠️  Warning: {e}')
"

echo ""
echo "🎉 Setup complete! You can now run: streamlit run streamlit_app.py"
