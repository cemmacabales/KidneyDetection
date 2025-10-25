#!/usr/bin/env python3
"""
Test script to verify ultralytics installation
"""

def test_ultralytics_import():
    """Test if ultralytics can be imported successfully"""
    try:
        from ultralytics import YOLO
        print("✅ ultralytics imported successfully!")
        
        # Try to get version info
        try:
            import ultralytics
            if hasattr(ultralytics, '__version__'):
                print(f"📦 ultralytics version: {ultralytics.__version__}")
            else:
                print("📦 ultralytics version: unknown")
        except:
            print("📦 ultralytics version: could not determine")
            
        # Test YOLO class
        print("🔍 Testing YOLO class...")
        yolo_class = YOLO
        print(f"✅ YOLO class available: {yolo_class}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import ultralytics: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_torch_import():
    """Test if torch can be imported successfully"""
    try:
        import torch
        print(f"✅ torch imported successfully! Version: {torch.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import torch: {e}")
        return False

def test_dependencies():
    """Test all major dependencies"""
    print("🧪 Testing dependencies...")
    print("-" * 50)
    
    dependencies = [
        ("numpy", "import numpy"),
        ("PIL", "from PIL import Image"),
        ("streamlit", "import streamlit"),
        ("pandas", "import pandas"),
    ]
    
    for name, import_cmd in dependencies:
        try:
            exec(import_cmd)
            print(f"✅ {name}: OK")
        except ImportError as e:
            print(f"❌ {name}: FAILED - {e}")

if __name__ == "__main__":
    print("🔬 Ultralytics Installation Test")
    print("=" * 50)
    
    # Test dependencies first
    test_dependencies()
    print()
    
    # Test torch
    torch_ok = test_torch_import()
    print()
    
    # Test ultralytics
    ultralytics_ok = test_ultralytics_import()
    print()
    
    if torch_ok and ultralytics_ok:
        print("🎉 All tests passed! Your environment should work with Streamlit.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("💡 Try running: pip install -r requirements.txt")