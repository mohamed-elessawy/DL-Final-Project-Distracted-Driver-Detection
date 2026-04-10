import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os

# 1. Define the DriverCNN Multimodal Architecture
class DriverCNNMultimodal(nn.Module):
    def __init__(self, num_classes=10, yolo_feat_dim=51):
        super().__init__()
        
        # ── CNN backbone (same blocks as DriverCNN) ──
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25)
        )
        self.cnn_pool = nn.AdaptiveAvgPool2d((4, 4))  # → (128*4*4) = 2048
        
        # ── YOLO MLP branch ──
        self.yolo_mlp = nn.Sequential(
            nn.Linear(yolo_feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        
        # ── Fusion classifier ──
        cnn_out_dim = 128 * 4 * 4  # 2048
        yolo_out_dim = 64
        fused_dim = cnn_out_dim + yolo_out_dim  # 2112
        
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_yolo):
        # CNN path
        x = self.block1(x_img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.cnn_pool(x)
        x = x.flatten(1)               # (B, 2048)
        
        # YOLO path
        y = self.yolo_mlp(x_yolo)      # (B, 64)
        
        # Fuse & classify
        fused = torch.cat([x, y], dim=1)  # (B, 2112)
        return self.classifier(fused)

# 2. Define the YOLO Feature Extraction Function
def extract_yolo_features(img_bgr, yolo_model, num_keypoints=17):
    """Extracts and normalizes the 17 keypoints from a live frame."""
    results = yolo_model(img_bgr, conf=0.4, verbose=False)
    kp_data = results[0].keypoints
    
    if kp_data is not None and len(kp_data.data) > 0:
        keypoints = kp_data.data[0].cpu().numpy()  # (17, 3)
        h, w = img_bgr.shape[:2]
        keypoints[:, 0] /= w  # Normalize x
        keypoints[:, 1] /= h  # Normalize y
        return keypoints.flatten()  # Returns (51,)
    else:
        return np.zeros(num_keypoints * 3, dtype=np.float32)

# 3. Class Map Mapping
class_map = {
    0: 'Normal Driving', 1: 'Texting - Right', 2: 'Talking on Phone - Right',
    3: 'Texting - Left', 4: 'Talking on Phone - Left', 5: 'Operating Radio',
    6: 'Drinking', 7: 'Reaching Behind', 8: 'Hair and Makeup', 9: 'Talking to Passenger'
}

# 4. Setup Device and Load Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading models on: {device}")

# Load YOLO
yolo_path = 'yolov8n-pose.pt'
yolo_model = YOLO(yolo_path)

# Load DriverCNN Multimodal
mm_model = DriverCNNMultimodal(num_classes=10).to(device)
weights_path = 'Models/best_model_MM.pth'
mm_model.load_state_dict(torch.load(weights_path, map_location=device))
mm_model.eval()

# 5. Define Image Transformations (1-Channel Grayscale ONLY)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 6. Initialize Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # --- Step 1: Extract YOLO Features ---
    yolo_feats = extract_yolo_features(frame, yolo_model)
    yolo_tensor = torch.tensor(yolo_feats, dtype=torch.float32).unsqueeze(0).to(device)

    # --- Step 2: Prepare Image Tensor ---
    # Convert OpenCV BGR to RGB, then apply Grayscale/Resize transforms
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    img_tensor = transform(pil_image).unsqueeze(0).to(device)

    # --- Step 3: Run Multimodal Inference ---
    with torch.no_grad():
        outputs = mm_model(img_tensor, yolo_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
        label_idx = predicted_class.item()
        confidence_score = confidence.item() * 100
        predicted_label = class_map[label_idx]

    # --- Step 4: Display Results ---
    text = f"{predicted_label} ({confidence_score:.1f}%)"
    color = (0, 255, 0) if label_idx == 0 else (0, 0, 255)
    
    # We plot the YOLO skeleton so you can visualize the geometric features being extracted
    results = yolo_model(frame, conf=0.4, verbose=False)
    annotated_frame = results[0].plot()
    
    cv2.putText(annotated_frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    cv2.imshow('DriverCNN Multimodal Live Test', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()