import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

# ==========================================
# Model Architecture & Functions
# ==========================================
class EfficientNetMultimodal(nn.Module):
    def __init__(self, num_classes=10, yolo_feat_dim=51):
        super(EfficientNetMultimodal, self).__init__()
        
        self.backbone = models.efficientnet_b3(weights=None)
        cnn_out_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.yolo_mlp = nn.Sequential(
            nn.Linear(yolo_feat_dim, 128),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(inplace=True)
        )
        
        fused_dim = cnn_out_dim + 64
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_yolo):
        if x_img.size(1) == 1:
            x_img = x_img.repeat(1, 3, 1, 1)
        x_cnn = self.backbone(x_img)
        x_yolo_out = self.yolo_mlp(x_yolo)
        fused = torch.cat([x_cnn, x_yolo_out], dim=1)
        return self.classifier(fused)

def extract_yolo_features(img_bgr, yolo_model, num_keypoints=17):
    results = yolo_model(img_bgr, conf=0.1, verbose=False)
    kp_data = results[0].keypoints
    if kp_data is not None and len(kp_data.data) > 0:
        keypoints = kp_data.data[0].cpu().numpy()
        h, w = img_bgr.shape[:2]
        keypoints[:, 0] /= w
        keypoints[:, 1] /= h
        return keypoints.flatten()
    return np.zeros(num_keypoints * 3, dtype=np.float32)

class_map = {
    0: 'Normal Driving', 1: 'Texting - Right', 2: 'Talking on Phone - Right',
    3: 'Texting - Left', 4: 'Talking on Phone - Left', 5: 'Operating Radio',
    6: 'Drinking', 7: 'Reaching Behind', 8: 'Hair and Makeup', 9: 'Talking to Passenger'
}