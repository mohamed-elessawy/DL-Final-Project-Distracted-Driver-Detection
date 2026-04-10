import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
import tempfile
import time

# Import your custom backend logic
from inference import EfficientNetMultimodal, extract_yolo_features, class_map

# ==========================================
# 1. Streamlit Setup & Custom CSS
# ==========================================
st.set_page_config(page_title="Distracted Driver Detection", layout="wide")

st.markdown("""
<style>
    /* Main typography and colors */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f7f9fa;
        border-right: 1px solid #e1e4e8;
    }
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #2980b9;
    }
    /* Subtitle styling */
    .subtitle {
        font-size: 1.1rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    /* Instruction container */
    .instruction-box {
        background-color: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Model Loading & Caching
# ==========================================
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo = YOLO('../models/yolov8n-pose.pt')
    effnet = EfficientNetMultimodal(num_classes=10).to(device)
    
    weights_path = '../models/best_model_Effnet.pth'
    if not os.path.exists(weights_path):
        st.error(f"System Error: Model weights not found at {weights_path}.")
        st.stop()
        
    effnet.load_state_dict(torch.load(weights_path, map_location=device))
    effnet.eval()
    return yolo, effnet, device

yolo_model, effnet_model, device = load_models()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ==========================================
# 3. Core Inference Function
# ==========================================
def process_frame(img_bgr):
    # 1. YOLO Features
    yolo_feats = extract_yolo_features(img_bgr, yolo_model)
    yolo_tensor = torch.tensor(yolo_feats, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 2. Image Tensor
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # 3. Predict
    with torch.no_grad():
        outputs = effnet_model(img_tensor, yolo_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
    label_idx = predicted_class.item()
    conf_score = confidence.item() * 100
    prediction_text = class_map[label_idx]
    
    # 4. Annotate image (boxes=False removes the messy bounding box and label)
    results = yolo_model(img_bgr, conf=0.1, verbose=False)
    annotated_bgr = results[0].plot(boxes=False)
    
    color = (0, 255, 0) if label_idx == 0 else (0, 0, 255)
    text = f"{prediction_text} ({conf_score:.1f}%)"
    cv2.putText(annotated_bgr, text, (10, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)
    
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, prediction_text, conf_score, label_idx

# ==========================================
# 4. Dashboard UI
# ==========================================
st.title("Distracted Driver Detection System")
st.markdown('<p class="subtitle">Multimodal Fusion Architecture (EfficientNet-B3 & YOLOv8-Pose)</p>', unsafe_allow_html=True)

# System Instructions
with st.expander("System Overview & Instructions", expanded=False):
    st.markdown("""
    <div class="instruction-box">
        <strong>Overview:</strong> This system utilizes a dual-branch neural network. It processes standard visual data through an EfficientNet-B3 backbone while simultaneously extracting and analyzing the driver's skeletal geometry via YOLOv8.
        <br><br>
        <strong>Instructions:</strong>
        <ol>
            <li>Select your input method from the sidebar on the left.</li>
            <li>For <strong>Static Images</strong>, upload one or multiple files. If multiple are uploaded, a slider will appear to navigate the gallery.</li>
            <li>For <strong>Video/Live Feed</strong>, the system will process the data frame-by-frame. Performance metrics (FPS) will be displayed.</li>
            <li><em>Tip: To view the video in fullscreen, hover over the top-right corner of the video player and click the expand icon.</em></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("Control Panel")
st.sidebar.markdown("---")
st.sidebar.subheader("Input Selection")
input_option = st.sidebar.radio(
    "Data Source:", 
    ("Image Upload", "Camera Snapshot", "Video Upload (MP4)", "Live Camera Feed")
)
st.sidebar.markdown("---")

# ---------------------------------------------------------
# Mode 1 & 2: Static Images (With Multi-File Support)
# ---------------------------------------------------------
if input_option in ["Image Upload", "Camera Snapshot"]:
    image_data = None
    
    if input_option == "Image Upload":
        # Enable multiple files
        uploaded_files = st.sidebar.file_uploader("Select Image Files (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded_files:
            # If multiple files exist, render a gallery slider
            if len(uploaded_files) > 1:
                st.sidebar.markdown("---")
                st.sidebar.markdown(f"**Gallery: {len(uploaded_files)} Images**")
                img_idx = st.sidebar.slider("Navigate Gallery", 1, len(uploaded_files), 1) - 1
            else:
                img_idx = 0
                
            image_data = uploaded_files[img_idx].getvalue()
            st.sidebar.info(f"Viewing File: {uploaded_files[img_idx].name}")
            
    elif input_option == "Camera Snapshot":
        camera_file = st.camera_input("Capture Image")
        if camera_file:
            image_data = camera_file.getvalue()

    if image_data is not None:
        col1, col2 = st.columns([2, 1])
        
        with st.spinner("Processing image..."):
            nparr = np.frombuffer(image_data, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            annotated_img, prediction, confidence, label_idx = process_frame(img_bgr)
        
        with col1:
            st.image(annotated_img, caption="Spatial & Geometric Analysis", use_container_width=True)
            
        with col2:
            st.subheader("Classification Results")
            if label_idx == 0:
                st.success(f"Classification: {prediction}")
            else:
                st.error(f"Classification: {prediction}")
            
            st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
    else:
        st.info("Awaiting input. Please provide an image via the control panel.")

# ---------------------------------------------------------
# Mode 3: Upload Video (MP4)
# ---------------------------------------------------------
elif input_option == "Video Upload (MP4)":
    uploaded_video = st.sidebar.file_uploader("Select Video File (MP4/AVI)", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        st.info("Initializing frame-by-frame analysis. Hover over the video feed to access the fullscreen button.")
        
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            stframe = st.empty()
        with col2:
            st.markdown("### Real-Time Metrics")
            fps_text = st.empty()
            status_text = st.empty()
        
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
                
            annotated_frame, pred, conf, label = process_frame(frame)
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)
            
            # FPS Calculation
            fps = 1.0 / (time.time() - start_time)
            fps_text.metric("Processing Speed", f"{fps:.1f} FPS")
            
            if label == 0:
                status_text.success(f"{pred} ({conf:.1f}%)")
            else:
                status_text.error(f"{pred} ({conf:.1f}%)")
            
        cap.release()
        st.success("Video analysis complete.")

# ---------------------------------------------------------
# Mode 4: Live Camera Feed
# ---------------------------------------------------------
elif input_option == "Live Camera Feed":
    st.sidebar.info("Enable the toggle below to activate the hardware camera.")
    run_camera = st.sidebar.checkbox("Activate Camera")
    
    if run_camera:
        st.info("Live feed active. Hover over the feed to access the fullscreen button.")
        cap = cv2.VideoCapture(0)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            stframe = st.empty()
        with col2:
            st.markdown("### Real-Time Metrics")
            fps_text = st.empty()
            status_text = st.empty()
        
        if not cap.isOpened():
            st.error("Hardware error: Could not establish connection with the camera.")
        else:
            while run_camera:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    st.error("Feed interrupted: Failed to capture frame.")
                    break
                    
                annotated_frame, pred, conf, label = process_frame(frame)
                stframe.image(annotated_frame, channels="RGB", use_container_width=True)
                
                # FPS Calculation
                fps = 1.0 / (time.time() - start_time)
                fps_text.metric("Processing Speed", f"{fps:.1f} FPS")
                
                if label == 0:
                    status_text.success(f"{pred} ({conf:.1f}%)")
                else:
                    status_text.error(f"{pred} ({conf:.1f}%)")
                    
            cap.release()
    else:
        st.info("Camera inactive. Toggle 'Activate Camera' in the control panel to begin.")