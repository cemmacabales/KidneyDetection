# Centralized table values for the Dataset page.
# Edit these lists to update what appears in the app.

# Tab 1: Dataset Preparation

dataset_summary = [
    {"Category": "Total Images", "Count/Details": "14,761", "Notes": "8,514 (Axial) + 6,247 (Coronal)"},
    {"Category": "Training Set", "Count/Details": "12,924", "Notes": "7,455 (Axial) + 5,469 (Coronal)"},
    {"Category": "Validation Set", "Count/Details": "612", "Notes": "353 (Axial) + 259 (Coronal)"},
    {"Category": "Test Set", "Count/Details": "1,225", "Notes": "706 (Axial) + 519 (Coronal)"},
]

view_distribution = [
    {"View Type": "Coronal View", "Count": "6,247", "Percentage": "42.4%", "Description": "Front-to-back kidney perspective"},
    {"View Type": "Axial View", "Count": "8,514", "Percentage": "57.7%", "Description": "Cross-sectional kidney slices"},
]

pathology_distribution = [
    {"Pathology Type": "Cysts", "Training Count": "3,636", "Validation Count": "170", "Test Count": "341", "Severity Levels": "Mild/Moderate/Severe"},
    {"Pathology Type": "Stones", "Training Count": "4,500", "Validation Count": "214", "Test Count": "428", "Severity Levels": "Simple/Complex"},
    {"Pathology Type": "Tumors", "Training Count": "4,788", "Validation Count": "228", "Test Count": "456", "Severity Levels": "Benign/Malignant"},
]

# Tab 2: Dataset Configuration

config_data = [
    {"Configuration Item": "Train/Val/Test Split", "Value": "70/10/20", "Rationale": "Standard ML practice"},
    {"Configuration Item": "Cross-Validation Folds", "Value": "", "Rationale": "Robust evaluation"},
    {"Configuration Item": "Stratification Method", "Value": "", "Rationale": "Balanced distribution"},
    {"Configuration Item": "Random Seed", "Value": "", "Rationale": "Reproducibility"},
    {"Configuration Item": "Batch Size", "Value": "", "Rationale": "Memory optimization"},
    {"Configuration Item": "Data Loading", "Value": "", "Rationale": "Efficient processing"},
]

# Tab 3: Preprocessing Pipeline & Data Augmentation

# Preprocessing timeline (single step as requested)
preprocessing_steps = [
    { "Step": "Image Resize", "Process": "Image Scaling", "Parameters": "640x640", "Output": "Uniform image shape"},
]

# Data Augmentation settings (key-value table)
augmentation_settings = [
    {"Setting": "Outputs per training example", "Value": "x3"},
    {"Setting": "Brightness", "Value": "Between -25% and +25%"},
    {"Setting": "Blur", "Value": "Up to 0.8px"},
    {"Setting": "Noise", "Value": "Up to 0.54% of pixels"},
]