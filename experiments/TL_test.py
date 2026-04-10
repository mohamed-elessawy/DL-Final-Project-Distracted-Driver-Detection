import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os

# 1. Define the Multimodal Architecture
class EfficientNetMultimodal(nn.Module):
    def __init__(self, num_classes=10, yolo_feat_dim=51):
        super(EfficientNetMultimodal, self).__init__()
        
        # ── 1. EfficientNet-B3 Backbone ──
        self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        
        # Extract the output dimension of the backbone (1536 for B3) before replacing the classifier
        cnn_out_dim = self.backbone.classifier[1].in_features
        
        # Remove the original classification head
        self.backbone.classifier = nn.Identity()
        
        # Freeze the entire backbone for Phase 1
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # ── 2. YOLO MLP Branch ──
        self.yolo_mlp = nn.Sequential(
            nn.Linear(yolo_feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        
        # ── 3. Fusion Classifier ──
        fused_dim = cnn_out_dim + 64  # 1536 + 64 = 1600
        
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
        # Dynamically adapt Grayscale (1-channel) to RGB (3-channel)
        if x_img.size(1) == 1:
            x_img = x_img.repeat(1, 3, 1, 1)
            
        # Extract visual features (B, 1536)
        x_cnn = self.backbone(x_img)
        
        # Extract geometric features (B, 64)
        x_yolo_out = self.yolo_mlp(x_yolo)
        
        # Fuse and Classify
        fused = torch.cat([x_cnn, x_yolo_out], dim=1)
        return self.classifier(fused)

# 2. Define the YOLO Feature Extraction Function
def extract_yolo_features(img_bgr, yolo_model, num_keypoints=17):
    results = yolo_model(img_bgr, conf=0.4, verbose=False)
    
    # results[0].keypoints.data shape: (num_persons, 17, 3)
    kp_data = results[0].keypoints
    if kp_data is not None and len(kp_data.data) > 0:
        # Take the first detected person, flatten to (51,)
        keypoints = kp_data.data[0].cpu().numpy()  # (17, 3)
        # Normalize x,y to [0,1] by image dimensions
        h, w = img_bgr.shape[:2]
        keypoints[:, 0] /= w  # x
        keypoints[:, 1] /= h  # y
        return keypoints.flatten()  # (51,)
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
if not os.path.exists(yolo_path):
    print(f"Downloading {yolo_path}...")
yolo_model = YOLO(yolo_path)

# Load EfficientNet Multimodal
effnet_model = EfficientNetMultimodal(num_classes=10).to(device)
weights_path = 'Models/best_model_Effnet.pth'
effnet_model.load_state_dict(torch.load(weights_path, map_location=device))
effnet_model.eval()

# 5. Define Image Transformations
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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    img_tensor = transform(pil_image).unsqueeze(0).to(device)

    # --- Step 3: Run Multimodal Inference ---
    with torch.no_grad():
        outputs = effnet_model(img_tensor, yolo_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
        label_idx = predicted_class.item()
        confidence_score = confidence.item() * 100
        predicted_label = class_map[label_idx]

    # --- Step 4: Display Results ---
    text = f"{predicted_label} ({confidence_score:.1f}%)"
    color = (0, 255, 0) if label_idx == 0 else (0, 0, 255)
    
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    # Optional: Draw YOLO skeleton on the frame for visual feedback
    results = yolo_model(frame, conf=0.4, verbose=False)
    annotated_frame = results[0].plot()
    
    cv2.putText(annotated_frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    cv2.imshow('EfficientNet Multimodal Live Test', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()