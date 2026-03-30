# Distracted Driver Detection Streamlit App

## Overview
This is a Streamlit web application for detecting distracted driving behaviors using a deep learning model. The app leverages a pre-trained model (`best_model.pth`) and YOLOv8 pose estimation (`yolov8n-pose.pt`) to analyze uploaded images or video frames for driver distraction (e.g., phone use, eating, etc.).

Key components:
- **app.py**: Main Streamlit interface for uploading media and getting predictions.
- **best_model.pth**: Trained classification model for distraction detection.
- **yolov8n-pose.pt**: YOLOv8 nano pose estimation model for keypoint detection.

## Prerequisites
- Python 3.8+
- Streamlit
- PyTorch
- Ultralytics (for YOLOv8)
- OpenCV

## RUN 
4. Ensure model files are present:
   - `best_model.pth`
   - `yolov8n-pose.pt`

## Usage
1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open the provided local URL (e.g., http://localhost:8501) in your browser.

3. Upload an image or video file of a driver.

4. The app will:
   - Detect driver pose using YOLOv8.
   - Classify distraction level using the best model.
   - Display results with visualizations.

## Model Details
- **Input**: Images/videos of drivers (State Farm Distracted Driver Dataset format assumed `images are taken side view`).
- **Output**: Distraction class (e.g., 'Normal Driving',
  'Texting - Right',
 'Talking on Phone - Right', etc.) with confidence scores.

