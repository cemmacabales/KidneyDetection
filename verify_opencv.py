#!/usr/bin/env python3
"""
Verify OpenCV Installation for Cloud Deployment
This script checks if opencv-python-headless is correctly installed
and that opencv-python (with GUI dependencies) is not present.
"""

import sys

def check_opencv_installation():
    """Check OpenCV installation and provide detailed diagnostics."""
    print("🔍 Checking OpenCV installation...\n")
    
    # Check if cv2 can be imported
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ Failed to import OpenCV: {e}")
        return False
    
    # Check build information for GUI dependencies
    build_info = cv2.getBuildInformation()
    
    # Check for GUI-related flags
    has_gtk = "WITH_GTK" in build_info and "GTK:                         YES" in build_info
    has_qt = "WITH_QT" in build_info and "QT:                          YES" in build_info
    has_gui = has_gtk or has_qt
    
    print(f"GUI support (GTK): {'❌ YES (problematic)' if has_gtk else '✅ NO (good)'}")
    print(f"GUI support (QT): {'❌ YES (problematic)' if has_qt else '✅ NO (good)'}")
    
    # Check which OpenCV package is installed
    try:
        import pkg_resources
        opencv_packages = []
        for pkg in pkg_resources.working_set:
            if 'opencv' in pkg.key.lower():
                opencv_packages.append(f"{pkg.key}=={pkg.version}")
        
        print(f"\n📦 Installed OpenCV packages:")
        for pkg in opencv_packages:
            if 'opencv-python-headless' in pkg:
                print(f"  ✅ {pkg}")
            else:
                print(f"  ⚠️  {pkg}")
        
        # Check for problematic packages
        problematic = [p for p in opencv_packages if 'opencv-python-headless' not in p]
        if problematic:
            print(f"\n⚠️  Warning: Found non-headless OpenCV packages: {problematic}")
            print("   These may cause issues in cloud deployment!")
            print("   Run: pip uninstall opencv-python opencv-contrib-python")
            print("   Then: pip install opencv-python-headless")
            
    except ImportError:
        print("⚠️  Could not check installed packages (pkg_resources not available)")
    
    # Check ultralytics compatibility
    print("\n🔍 Checking Ultralytics compatibility...")
    try:
        import ultralytics
        print(f"✅ Ultralytics version: {ultralytics.__version__}")
        
        # Try to create a simple YOLO instance
        try:
            from ultralytics import YOLO
            print("✅ YOLO can be imported successfully")
        except Exception as e:
            error_msg = str(e)
            if "libGL.so.1" in error_msg or "OpenGL" in error_msg:
                print(f"❌ OpenGL/libGL error detected: {error_msg}")
                print("   This suggests opencv-python (full version) might be installed")
                print("   or system GL libraries are missing.")
                return False
            else:
                print(f"⚠️  YOLO import warning: {error_msg}")
    except ImportError as e:
        print(f"⚠️  Ultralytics not installed: {e}")
    
    # Final verdict
    print("\n" + "="*60)
    if has_gui:
        print("❌ DEPLOYMENT CHECK FAILED")
        print("   opencv-python (with GUI) is installed instead of opencv-python-headless")
        print("   This will cause issues in cloud environments without display servers")
        print("\n   FIX:")
        print("   1. pip uninstall opencv-python opencv-contrib-python")
        print("   2. pip install opencv-python-headless")
        print("   3. Run this script again to verify")
        return False
    else:
        print("✅ DEPLOYMENT CHECK PASSED")
        print("   opencv-python-headless is correctly installed")
        print("   Your application is ready for cloud deployment!")
        return True

if __name__ == "__main__":
    success = check_opencv_installation()
    sys.exit(0 if success else 1)
