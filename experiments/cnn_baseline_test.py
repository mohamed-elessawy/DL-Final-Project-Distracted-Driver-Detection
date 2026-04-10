import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. Define the exact DriverCNN architecture
class DriverCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DriverCNN, self).__init__()

        # --- Block 1: 1 → 32 ---   
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),   # 224×224×32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 112×112×32
            nn.Dropout2d(0.25)
        )

        # --- Block 2: 32 → 64 ---
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 112×112×64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 56×56×64
            nn.Dropout2d(0.25)
        )

        # --- Block 3: 64 → 128 ---
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=0), # 56×56×128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 28×28×128
            nn.Dropout2d(0.25)
        )

        # --- Block 4: 128 → 256 ---
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# 28×28×256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 14×14×256
            nn.Dropout2d(0.25)
        )

        # --- Classifier head ---
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),                  # 4×4×256 = 4096
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)                    # 10 classes
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

# 2. Class Map Mapping
class_map = {
    0: 'Normal Driving',
    1: 'Texting - Right',
    2: 'Talking on Phone - Right',
    3: 'Texting - Left',
    4: 'Talking on Phone - Left',
    5: 'Operating Radio',
    6: 'Drinking',
    7: 'Reaching Behind',
    8: 'Hair and Makeup',
    9: 'Talking to Passenger'
}

# 3. Setup Device and Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading model on: {device}")

model = DriverCNN(num_classes=10).to(device)

# Load the weights from the Models folder
weights_path = 'Models/best_model_CNN.pth'
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# 4. Define Image Transformations (Must match training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 5. Initialize Webcam
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

    # Convert frame for PyTorch (OpenCV uses BGR, PIL uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Apply transforms and add batch dimension: (1, 1, 256, 256)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # 6. Run Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
        label_idx = predicted_class.item()
        confidence_score = confidence.item() * 100
        predicted_label = class_map[label_idx]

    # 7. Display Results on the Frame
    text = f"{predicted_label} ({confidence_score:.1f}%)"
    
    # Change color to red if distracted, green if normal driving
    color = (0, 255, 0) if label_idx == 0 else (0, 0, 255)
    
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Show the video fee
    cv2.imshow('DriverCNN Live Test', frame)

    # Quit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()