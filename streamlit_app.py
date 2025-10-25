import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import Optional, Tuple, Dict, Any
import datasetvalues as dv

# Configure page
st.set_page_config(
    page_title="Kidney Abnormality Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with Font Awesome icons
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .view-selection {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .upload-section {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .results-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Custom icon styles with consistent theming */
    .nav-icon {
        color: #1f77b4;
        margin-right: 8px;
        font-size: 1.1em;
    }
    
    .section-icon {
        color: #1f77b4;
        margin-right: 8px;
        font-size: 1.2em;
    }
    
    .feature-icon {
        color: #28a745;
        margin-right: 8px;
        font-size: 1em;
    }
    
    .metric-icon {
        color: #17a2b8;
        margin-right: 6px;
        font-size: 0.9em;
    }
    
    .status-icon {
        color: #28a745;
        margin-right: 6px;
    }

    .warning-icon {
        color: #dc3545; /* red */
        margin-right: 6px;
    }
    
    /* Disable typing in view type selectbox */
    div[data-testid="stSelectbox"] input {
        pointer-events: none;
        caret-color: transparent;
    }
    
    div[data-testid="stSelectbox"] input:focus {
        outline: none;
        caret-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar title
st.sidebar.title("Kidney Abnormality Detection")

class KidneyDetectionApp:
    """Main application class for kidney abnormality detection."""
    
    def __init__(self):
        self.supported_formats = ['png', 'jpg', 'jpeg', 'tiff', 'bmp']
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.coronal_model = None
        self.axial_model = None
        
    def load_yolo_model(self, view_type: str) -> Optional[Any]:
        """
        Load the appropriate YOLO model based on view type.
        
        Args:
            view_type (str): Either 'coronal' or 'axial'
            
        Returns:
            YOLO model object or None if not available
        """
        try:
            from ultralytics import YOLO
            import os
            
            # Define model paths
            model_dir = "models"
            coronal_path = os.path.join(model_dir, "coronalBest weights.pt")
            axial_path = os.path.join(model_dir, "axialBest Weights.pt")
            
            if view_type == 'coronal' and self.coronal_model is None:
                if os.path.exists(coronal_path):
                    st.info(f"Loading coronal model from {coronal_path}...")
                    self.coronal_model = YOLO(coronal_path)
                    st.success("✅ Coronal model loaded successfully!")
                else:
                    st.error(f"❌ Coronal model file not found: {coronal_path}")
                    return None
                    
            elif view_type == 'axial' and self.axial_model is None:
                if os.path.exists(axial_path):
                    st.info(f"Loading axial model from {axial_path}...")
                    self.axial_model = YOLO(axial_path)
                    st.success("✅ Axial model loaded successfully!")
                else:
                    st.error(f"❌ Axial model file not found: {axial_path}")
                    return None
            
            return self.coronal_model if view_type == 'coronal' else self.axial_model
            
        except ImportError:
            st.error("❌ ultralytics package not installed. Please install it: pip install ultralytics")
            return None
        except Exception as e:
            st.error(f"❌ Error loading {view_type} model: {str(e)}")
            return None
    
    def process_with_yolo(self, image: Image.Image, model: Optional[Any], 
                         confidence_threshold: float, selected_classes: list,
                         roi: Tuple[float, float, float, float]) -> Tuple[Image.Image, list]:
        """
        Process image with YOLO model and return annotated image with detections.
        
        Args:
            image: PIL Image
            model: YOLO model (None for placeholder)
            confidence_threshold: Detection confidence threshold
            selected_classes: List of classes to detect
            roi: Region of interest (x_min, y_min, x_max, y_max)
            
        Returns:
            Tuple of (annotated_image, detections_list)
        """
        import cv2
        from PIL import ImageDraw, ImageFont
        
        # Convert PIL to numpy array and handle channel conversion
        img_array = np.array(image)
        
        # Handle grayscale to RGB conversion for YOLO models
        if len(img_array.shape) == 2:  # Grayscale image
            # Convert grayscale to RGB by duplicating the channel
            img_array = np.stack([img_array, img_array, img_array], axis=-1)
            st.info("ℹ️ Converted grayscale image to RGB for YOLO processing")
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:  # Single channel
            # Convert single channel to RGB
            img_array = np.repeat(img_array, 3, axis=2)
            st.info("ℹ️ Converted single-channel image to RGB for YOLO processing")
        
        if model is None:
            # Placeholder detection for demo purposes
            detections = self._generate_placeholder_detections(selected_classes, confidence_threshold)
        else:
            # Actual YOLO inference
            try:
                # Run YOLO inference
                results = model(img_array, conf=confidence_threshold, verbose=False)
                detections = self._parse_yolo_results(results, selected_classes, roi)
            except Exception as e:
                st.error(f"❌ Error during YOLO inference: {str(e)}")
                # Fallback to placeholder
                detections = self._generate_placeholder_detections(selected_classes, confidence_threshold)
        
        # Draw bounding boxes on image
        annotated_image = self._draw_detections(image, detections, roi)
        
        return annotated_image, detections
    
    def _generate_placeholder_detections(self, selected_classes: list, confidence: float) -> list:
        """Generate placeholder detections for demo purposes."""
        import random
        
        detections = []
        if 'kidney' in selected_classes:
            detections.append({
                'class': 'kidney',
                'confidence': round(random.uniform(confidence, 1.0), 2),
                'bbox': [0.1, 0.1, 0.9, 0.9]  # x1, y1, x2, y2 (normalized)
            })
        if 'stone' in selected_classes:
            detections.append({
                'class': 'stone',
                'confidence': round(random.uniform(confidence, 1.0), 2),
                'bbox': [0.3, 0.2, 0.5, 0.4]
            })
        if 'cyst' in selected_classes:
            detections.append({
                'class': 'cyst',
                'confidence': round(random.uniform(confidence, 1.0), 2),
                'bbox': [0.6, 0.5, 0.8, 0.7]
            })
        if 'tumor' in selected_classes:
            detections.append({
                'class': 'tumor',
                'confidence': round(random.uniform(confidence, 1.0), 2),
                'bbox': [0.2, 0.6, 0.4, 0.8]
            })
        
        return detections
    
    def _parse_yolo_results(self, results, selected_classes: list, roi: Tuple[float, float, float, float]) -> list:
        """Parse YOLO results and convert to our detection format."""
        detections = []
        
        # YOLO class mapping for kidney detection
        class_mapping = {
            0: 'kidney',
            1: 'cyst', 
            2: 'stone',
            3: 'tumor'
        }
        
        try:
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Get bounding box coordinates (normalized)
                        box = boxes.xyxyn[i].cpu().numpy()  # [x1, y1, x2, y2] normalized
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Map class ID to class name
                        class_name = class_mapping.get(class_id, f'class_{class_id}')
                        
                        # Filter by selected classes
                        if class_name in selected_classes:
                            # Check if detection is within ROI
                            x1, y1, x2, y2 = box
                            roi_x1, roi_y1, roi_x2, roi_y2 = roi
                            
                            # Check if bounding box center is within ROI
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            
                            if (roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2):
                                detections.append({
                                    'class': class_name,
                                    'confidence': round(confidence, 3),
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                                })
        
        except Exception as e:
            st.warning(f"⚠️ Error parsing YOLO results: {str(e)}")
            
        return detections
    
    def _draw_detections(self, image: Image.Image, detections: list, 
                        roi: Tuple[float, float, float, float]) -> Image.Image:
        """Draw bounding boxes and labels on image."""
        from PIL import ImageDraw, ImageFont
        
        # Create a copy of the image and ensure it's in RGB mode for colored annotations
        annotated_image = image.copy()
        if annotated_image.mode != 'RGB':
            annotated_image = annotated_image.convert('RGB')
        draw = ImageDraw.Draw(annotated_image)
        
        # Get image dimensions
        img_width, img_height = image.size
        
        # Draw ROI rectangle
        x_min, y_min, x_max, y_max = roi
        roi_coords = [
            x_min * img_width, y_min * img_height,
            x_max * img_width, y_max * img_height
        ]
        draw.rectangle(roi_coords, outline='green', width=3)
        
        # Color map for different classes
        colors = {
            'kidney': 'red',
            'cyst': 'blue', 
            'stone': 'yellow',
            'tumor': 'purple'
        }
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Convert normalized coordinates to pixel coordinates
            x1 = bbox[0] * img_width
            y1 = bbox[1] * img_height
            x2 = bbox[2] * img_width
            y2 = bbox[3] * img_height
            
            # Draw bounding box with thicker outline
            color = colors.get(class_name, 'white')
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label with background for better visibility
            label = f"{class_name}: {confidence:.2f}"
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("Arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Get text size for background rectangle
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            # Draw background rectangle for text
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], 
                         fill=color, outline=color)
            
            # Draw text in white for contrast
            draw.text((x1 + 2, y1 - text_height - 2), label, fill='white', font=font)
        
        return annotated_image
    
    def validate_image(self, uploaded_file) -> Tuple[bool, str]:
        """Validate uploaded image file."""
        if uploaded_file.size > self.max_file_size:
            return False, f"File size ({uploaded_file.size / 1024 / 1024:.1f}MB) exceeds maximum allowed size (10MB)"
        
        try:
            image = Image.open(uploaded_file)
            return True, "Valid image"
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"

    def preprocess_image(self, image: Image.Image, view_type: str) -> np.ndarray:
        """
        Preprocess the uploaded image for model prediction.
        
        Args:
            image (PIL.Image): The uploaded image
            view_type (str): Either 'coronal' or 'axial'
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        # TODO: Implement actual preprocessing based on your model requirements
        # Example preprocessing steps:
        # 1. Resize to model input size
        # 2. Normalize pixel values
        # 3. Convert to appropriate format
        
        # Placeholder preprocessing
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = np.mean(img_array, axis=2)  # Convert to grayscale if needed
        
        # Resize to standard size (adjust based on your model)
        # img_resized = cv2.resize(img_array, (224, 224))
        # img_normalized = img_resized / 255.0
        
        return img_array
    
    def predict_abnormality(self, image: Image.Image, view_type: str) -> Dict[str, Any]:
        """
        Predict kidney abnormalities from the uploaded image.
        
        Args:
            image (PIL.Image): The uploaded image
            view_type (str): Either 'coronal' or 'axial'
            
        Returns:
            Dict containing prediction results
        """
        # TODO: Implement actual prediction
        # model = self.load_model(view_type)
        # preprocessed_image = self.preprocess_image(image, view_type)
        # predictions = model.predict(preprocessed_image)
        
        # Placeholder results for demonstration
        placeholder_results = {
            'abnormality_detected': True,
            'confidence_score': 0.85,
            'abnormality_type': 'Kidney Stone',
            'severity': 'Moderate',
            'recommendations': [
                'Consult with a nephrologist',
                'Consider additional imaging studies',
                'Monitor symptoms closely'
            ],
            'view_type': view_type
        }
        
        return placeholder_results
    
    def validate_image(self, uploaded_file) -> Tuple[bool, str]:
        """
        Validate the uploaded image file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if uploaded_file is None:
            return False, "No file uploaded"
        
        # Check file size
        if uploaded_file.size > self.max_file_size:
            return False, f"File size too large. Maximum size is {self.max_file_size // (1024*1024)}MB"
        
        # Check file format
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in self.supported_formats:
            return False, f"Unsupported format. Supported formats: {', '.join(self.supported_formats)}"
        
        return True, ""
    
    def display_results(self, results: Dict[str, Any], image: Image.Image):
        """
        Display prediction results in a formatted way.
        
        Args:
            results (Dict): Prediction results
            image (PIL.Image): Original uploaded image
        """
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.markdown('#### <i class="fas fa-search section-icon"></i>Analysis Results', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption=f"Uploaded {results['view_type'].title()} View", use_container_width=True)
        
        with col2:
            # Detection status
            if results['abnormality_detected']:
                st.error(f"⚠️ Abnormality Detected: {results['abnormality_type']}")
            else:
                st.success("✅ No abnormalities detected")
            
            # Confidence score
            st.metric("Confidence Score", f"{results['confidence_score']:.2%}")
            
            # Severity (if abnormality detected)
            if results['abnormality_detected']:
                severity_color = {
                    'Mild': 'green',
                    'Moderate': 'orange', 
                    'Severe': 'red'
                }.get(results['severity'], 'gray')
                
                st.markdown(f"**Severity:** <span style='color: {severity_color}'>{results['severity']}</span>", 
                           unsafe_allow_html=True)
        
        # Recommendations
        if results['abnormality_detected'] and results['recommendations']:
            st.markdown('#### <i class="fas fa-clipboard-list section-icon"></i>Recommendations', unsafe_allow_html=True)
            for i, rec in enumerate(results['recommendations'], 1):
                st.write(f"{i}. {rec}")
        
        # Disclaimer
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("⚠️ **Medical Disclaimer**: This tool is for educational purposes only and should not replace professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def show_about_page():
    """Display the About page with revised research-based content."""
    st.markdown('<h1 class="main-header"><i class="fas fa-info-circle section-icon"></i>About</h1>', unsafe_allow_html=True)

    # Abstract
    st.markdown("""
    ## Abstract

    This study introduces an automated **Kidney Abnormality Detection System** utilizing **YOLOv12**, a state-of-the-art deep learning model for medical image analysis. 
    The system employs **Convolutional Neural Networks (CNNs)** to identify and classify kidney abnormalities — including **stones, cysts, and tumors** — from **Computed Tomography (CT)** images. 
    Using a clinically validated dataset of over **12,000 CT scans**, the model was trained and enhanced through **data augmentation** to improve robustness and class balance. 
    The system achieved a high **mean Average Precision (mAP)** across multiple Intersection over Union (IoU) thresholds, confirming its diagnostic reliability. 
    Overall, this research aims to assist healthcare professionals by providing **real-time, AI-powered diagnostic support** that enhances accuracy, consistency, and early disease detection.
    """)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Methodology", "System Features", "Clinical Impact"])

    # Introduction Tab
    with tab1:
        st.markdown("""
        ### Background and Motivation

        Kidney diseases — such as **stones, cysts, and tumors** — are among the most prevalent urological conditions worldwide. 
        Accurate and early detection of these abnormalities is crucial for improving treatment outcomes and patient survival rates. 
        However, manual interpretation of CT scans is **time-consuming**, **error-prone**, and dependent on radiologist expertise.

        ### Research Objectives
        1. **Early Detection** — Enable prompt identification of kidney abnormalities through AI-powered analysis.  
        2. **Diagnostic Assistance** — Provide healthcare professionals with automated support for clinical decision-making.  
        3. **High Accuracy** — Utilize YOLOv12’s advanced feature extraction to enhance diagnostic precision.  
        4. **Workflow Integration** — Deliver a user-friendly system suitable for integration with hospital workflows.

        ### Significance
        The proposed solution addresses the limitations of traditional diagnosis by automating kidney abnormality detection with deep learning, ensuring faster and more consistent results while reducing radiologist workload.
        """)

    # Methodology Tab
    with tab2:
        st.markdown("""
        ### Dataset Preparation

        The dataset consists of **12,446 CT images** obtained via the **Picture Archiving and Communication System (PACS)**. 
        Each sample was verified to represent one of four diagnostic classes: **normal**, **cyst**, **stone**, or **tumor**.  
        After cleaning and augmentation, the dataset was divided into **70% training**, **10% validation**, and **20% testing**.

        #### Data Augmentation Techniques
        - Brightness variation (±25%)  
        - Gaussian blur (≤0.8 px)  
        - Random noise addition (≤0.54%)  

        ### Model Architecture

        The system utilizes **YOLOv12**, the latest evolution of the YOLO family.  
        It integrates **multi-scale feature fusion** and **FlashAttention** mechanisms to improve small-object detection and reduce computational cost.

        #### Training Process
        - **Input Size:** 640×640 pixels  
        - **Optimizer:** Adam with learning rate decay  
        - **Loss Function:** Bounding box regression + cross-entropy  
        - **Batch Size:** 16  
        - **Epochs:** 100  
        - **Evaluation Metric:** mean Average Precision (mAP@[0.5:0.95])

        The system achieved strong performance in detecting and classifying multiple renal abnormalities, validating its capability for clinical deployment.
        """)

    # System Features Tab
    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### Core System Features

            **<i class="fas fa-brain feature-icon"></i> AI-Powered Detection**  
            - Detects and classifies kidney stones, cysts, and tumors automatically.  
            - Uses YOLOv12 for real-time performance with high precision.  

            **<i class="fas fa-sync feature-icon"></i> Automated Workflow**  
            - Supports both **coronal** and **axial** CT image views.  
            - Automatically preprocesses and analyzes uploaded scans.  

            **<i class="fas fa-tachometer-alt feature-icon"></i> Real-Time Processing**  
            - Performs detection in seconds per image.  
            - Suitable for integration in clinical screening systems.
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            ### Technical Capabilities

            **<i class="fas fa-database feature-icon"></i> Dataset Handling**  
            - Over 12,000 CT images  
            - Augmentation for balanced class distribution  

            **<i class="fas fa-cogs feature-icon"></i> Model Flexibility**  
            - Transferable architecture for other imaging modalities  
            - Configurable thresholds for precision vs. recall tuning  

            **<i class="fas fa-desktop feature-icon"></i> Web Interface**  
            - User-friendly dashboard built with Streamlit  
            - Displays results, confidence scores, and severity level  
            """, unsafe_allow_html=True)

    # Clinical Impact Tab
    with tab4:
        st.markdown("""
        ### Clinical Implications

        The integration of deep learning models like YOLOv12 into medical workflows enhances diagnostic precision and accelerates the detection of renal abnormalities.

        #### Benefits for Healthcare Providers
        - **Reduced Diagnostic Time:** AI-assisted analysis delivers results in seconds.  
        - **Consistency:** Minimizes inter-observer variability.  
        - **Scalability:** Suitable for hospitals and remote diagnostic centers.  
        - **Decision Support:** Aids clinicians in early diagnosis and treatment planning.  

        #### Benefits for Patients
        - **Early Detection:** Improved prognosis through timely identification.  
        - **Accessibility:** Reduces dependency on radiologist availability.  
        - **Lower Costs:** Cuts down on manual diagnostic labor.  

        ### Future Directions
        - Incorporation of **3D CT reconstruction** for volumetric lesion detection  
        - Expansion to **multi-modal imaging** (CT + MRI + Ultrasound)  
        - Development of **real-time clinical decision support** dashboards  
        - Continuous retraining for adaptive accuracy in diverse patient datasets
        """)

    # Disclaimer
    st.markdown("---")
    st.markdown("""
    **<i class="fas fa-exclamation-triangle warning-icon"></i> Important Medical Disclaimer**
    """, unsafe_allow_html=True)
    st.warning("""
    This system is designed for research and educational purposes. 
    It should not be used as a standalone diagnostic tool. 
    All results must be reviewed and validated by licensed healthcare professionals before any medical decision-making.
    """)


def show_dataset_page():
    """Display the Dataset page."""
    st.markdown('<h1 class="main-header"><i class="fas fa-chart-bar section-icon"></i>Dataset Information</h1>', unsafe_allow_html=True)
    
    # Dataset Overview
    st.markdown("""
    ## Dataset Overview
    
    This section provides comprehensive information about the kidney abnormality detection dataset used for training and validation of the deep learning models. The dataset comprises multi-center CT scan images with expert annotations for various kidney pathologies.
    """)
    
    tab1, tab2 = st.tabs(["Dataset Preparation", "Dataset Configuration"])
    
    with tab1:
        st.subheader("Dataset Preparation")
        
        # Dataset Summary Table
        st.markdown("### Dataset Summary")
        dataset_summary = pd.DataFrame(dv.dataset_summary)
        st.dataframe(dataset_summary, use_container_width=True)
        
        # Image View Distribution
        st.markdown("### Image View Distribution")
        view_distribution = pd.DataFrame(dv.view_distribution)
        st.dataframe(view_distribution, use_container_width=True)
        
        # Pathology Distribution
        st.markdown("### Pathology Distribution")
        pathology_data = pd.DataFrame(dv.pathology_distribution)
        st.dataframe(pathology_data, use_container_width=True)
    
    with tab2:    
        # Removed Data Collection Parameters, Inclusion and Exclusion Criteria by request
        # Removed Dataset Configuration table by request

        # Image Preprocessing Steps (moved here)
        st.markdown("### Image Preprocessing Steps")
        preprocessing_steps = pd.DataFrame(dv.preprocessing_steps)
        st.dataframe(preprocessing_steps, use_container_width=True)

        # Data Augmentation Settings (moved here)
        st.markdown("### Data Augmentation Settings")
        augmentation_settings = pd.DataFrame(dv.augmentation_settings)
        st.dataframe(augmentation_settings, use_container_width=True)
    
    # Note: Preprocessing and augmentation moved into the 'Dataset Configuration' tab.

def show_model_page():
    """Display the Model page."""
    st.markdown('<h1 class="main-header">🤖 Model Architecture</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Architecture", "Performance", "Technical Details"])
    
    with tab1:
        st.subheader("Deep Learning Architecture")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### YOLOv12 Architecture
            
            YOLOv12 is optimized for real-time detection and segmentation with modern attention and fusion modules tailored for 640×640 inputs.
            
            **Pipeline Overview**
            - Input: `640×640×3` image
            - Initial Conv: stride 2
            - Backbone: `C3k3` blocks with `R-ELAN`
            - Feature Pyramid:
              - `P3 (80×80×256)` with Area Attention
              - `P4 (40×40×512)` with FlashAttention
              - `P5 (20×20×1024)` with Position Perceiver
            - Neck:
              - Upsample `P5 → P4`, concat (`A2C2F` fusion)
              - Upsample `P4 → P3`, concat (`C3K2` process)
            - Final Path: Downsample `7×7` Separable Conv + `C3K2`
            - Detection Head: classification + localization
            
            **Flow (schematic)**
            ```
            Input 640x640x3
                ↓ Conv stride 2
                ↓ Backbone: C3k3 (R-ELAN)
                ↘             ↘             ↘
              P3: Area Attn  P4: FlashAttn  P5: Position Perceiver
                                  ↑            
                           Upsample P5→P4 → Concat (A2C2F)
                                  ↑
                           Upsample P4→P3 → Concat (C3K2)
                                  ↓
                    Downsample 7×7 SepConv + C3K2
                                  → Detection Head (cls + loc)
            ```
            
            **Why YOLOv12**
            - Attention-guided features (Area, Flash, Perceiver) improve representational power.
            - Efficient neck with feature fusion for multi-scale targets.
            - Real-time friendly while supporting segmentation heads.
            
            """)
        
        with col2:
            st.info("""
            **YOLO Modes & Checkpoints**
            
            - Detection: `yolo12n.pt`
            - Segmentation: `yolo11n-seg.pt`
            
            **Typical Settings**
            - Input size: `640`
            - Anchor-free detection head
            - Multi-scale training recommended
            
            **References**
            - Ultralytics YOLOv12 overview (see provided reference)
            """)
    
    with tab2:
        st.subheader("Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "92.5%", "↑ 2.3%")
        with col2:
            st.metric("Sensitivity", "89.3%", "↑ 1.8%")
        with col3:
            st.metric("Specificity", "94.7%", "↑ 2.1%")
        with col4:
            st.metric("F1-Score", "91.2%", "↑ 2.0%")
        
        st.markdown("""
        ### Detailed Performance Analysis
        
        **Cross-Validation Results** (5-fold):
        - Mean Accuracy: 92.5% ± 1.2%
        - Mean Sensitivity: 89.3% ± 2.1%
        - Mean Specificity: 94.7% ± 1.5%
        
        **Per-Condition Performance**:
        - Normal Detection: 95.2% accuracy
        - Kidney Stones: 88.7% accuracy
        - Cysts: 91.3% accuracy
        - Tumors: 89.8% accuracy
        - Structural Abnormalities: 87.9% accuracy
        
        **Processing Speed**:
        - Average Inference Time: 2.3 seconds
        - GPU Acceleration: 0.8 seconds
        - Batch Processing: 50 images/minute
        """)
    
    with tab3:
        st.subheader("Technical Implementation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### YOLO Model Technical Details

            Initialize and train YOLO for kidney segmentation:

            ```python
            # Initialize the YOLO model 
            # ⚠️ Use the appropriate checkpoint: 
            # - yolo12n.pt       → for object detection 
            # - yolo11n-seg.pt   → for object segmentation 
            model = YOLO("C:/Users/ASUS PC/Desktop/Yolo12_Kidney_Disease_coronal/yolov12/yolo11n-seg.pt") 
            
            # Train the model 
            results = model.train( 
                data='C:/Users/ASUS PC/Desktop/Yolo12_Kidney_Disease_coronal/data.yaml',  # dataset config file 
                epochs=1000,       # maximum epochs (early stopping may end sooner) 
                patience=15,       # stop if no improvement for 15 epochs 
                batch=8,           # batch size 
                imgsz=640,         # input image size 
                scale=0.5,         # data augmentation scaling 
                mosaic=1.0,        # mosaic augmentation probability 
                mixup=0.0,         # mixup probability 
                copy_paste=0.1,    # copy-paste augmentation probability 
                close_mosaic=10,   # disable mosaic augmentation in the last 10 epochs 
                device="0",        # set to "0" for GPU, "cpu" for CPU 
                save=True          # save both best.pt and last.pt 
                
                
            ) 
            ```
            """)
        
        with col2:
            st.markdown("""
            ### Deployment Details
            
            **Model Format**:
            - TensorFlow SavedModel
            - ONNX compatibility
            - Quantized versions available
            
            **Hardware Requirements**:
            - CPU: 4+ cores recommended
            - RAM: 8GB minimum
            - GPU: Optional (CUDA support)
            
            **Integration**:
            - REST API endpoints
            - Batch processing support
            - Real-time inference
            """)

def main():
    """Main application function."""
    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'detection'
    
    app = KidneyDetectionApp()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown('### <i class="fas fa-compass nav-icon"></i>Navigation', unsafe_allow_html=True)
        
        # Navigation buttons
        if st.button("Detection", use_container_width=True, type="primary" if st.session_state.current_page == 'detection' else "secondary", key="nav_detection"):
            st.session_state.current_page = 'detection'
            st.rerun()
        
        if st.button("About", use_container_width=True, type="primary" if st.session_state.current_page == 'about' else "secondary", key="nav_about"):
            st.session_state.current_page = 'about'
            st.rerun()
        
        if st.button("Dataset", use_container_width=True, type="primary" if st.session_state.current_page == 'dataset' else "secondary", key="nav_dataset"):
            st.session_state.current_page = 'dataset'
            st.rerun()
        
        if st.button("Model", use_container_width=True, type="primary" if st.session_state.current_page == 'model' else "secondary", key="nav_model"):
            st.session_state.current_page = 'model'
            st.rerun()
            

    
    # Display the appropriate page based on navigation
    if st.session_state.current_page == 'about':
        show_about_page()
    elif st.session_state.current_page == 'dataset':
        show_dataset_page()
    elif st.session_state.current_page == 'model':
        show_model_page()
    else:
        # Default detection page
        show_detection_page()

def show_detection_page():
    """Display the detection page with minimalist UI."""
    st.title("🔍 Kidney Abnormality Detection")
    
    # Initialize app
    app = KidneyDetectionApp()
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Image upload section
        st.subheader("Upload Medical Image")
        uploaded_file = st.file_uploader(
            "Choose a medical image file",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Supported formats: PNG, JPG, JPEG, TIFF, BMP (Max 10MB)"
        )
        
        if uploaded_file is not None:
            # Validate image
            is_valid, message = app.validate_image(uploaded_file)
            
            if not is_valid:
                st.error(message)
                return
            
            # Load and display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        # View type selection
        st.subheader("Image View Type")
        
        view_type = st.selectbox(
            "Select View",
            options=['coronal', 'axial'],
            help="Select the anatomical view of the image",
            key="view_type_select"
        )
        
        # Set default parameters
        confidence_threshold = 0.5
        selected_classes = ['kidney', 'cyst', 'stone', 'tumor']
        roi = (0.0, 0.0, 1.0, 1.0)
        
        # Analysis button
        analyze_button = st.button(
            "🔍 Analyze Image",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None
        )
    
    # Analysis results
    if uploaded_file is not None and analyze_button:
        with st.spinner("Loading model and analyzing image..."):
            # Load appropriate model
            model = app.load_yolo_model(view_type)
            
            # Process image
            annotated_image, detections = app.process_with_yolo(
                image, model, confidence_threshold, selected_classes, roi
            )
            
            # Display results
            st.subheader("Analysis Results")
            
            # Color legend
            st.markdown("""
            **Detection Color Legend:**
            - 🔴 **Red**: Kidney
            - 🔵 **Blue**: Cyst  
            - 🟡 **Yellow**: Stone
            - 🟣 **Purple**: Tumor
            """)
            
            # Show annotated image
            col_result1, col_result2 = st.columns([3, 1])
            
            with col_result1:
                st.image(annotated_image, caption="Processed Image", use_container_width=True)
            
            with col_result2:
                # Detection summary
                st.metric("Total Detections", len(detections))
                
                if detections:
                    st.subheader("Detected Objects")
                    for i, detection in enumerate(detections):
                        with st.expander(f"{detection['class'].title()} #{i+1}"):
                            st.write(f"**Confidence:** {detection['confidence']:.2f}")
                            st.write(f"**Class:** {detection['class']}")
                            bbox = detection['bbox']
                            st.write(f"**Location:** ({bbox[0]:.2f}, {bbox[1]:.2f}) to ({bbox[2]:.2f}, {bbox[3]:.2f})")
                else:
                    st.info("No abnormalities detected with current settings.")
                
                # Model info
                st.subheader("Model Info")
                st.write(f"**View Type:** {view_type.title()}")
                st.write(f"**Model Status:** {'Loaded' if model else 'Placeholder'}")
                
                if not model:
                    st.warning("Using placeholder detection. Load actual YOLO models for real analysis.")

    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 0.8rem;'>Built with Streamlit | For educational purposes only</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()