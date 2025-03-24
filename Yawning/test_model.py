import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# Define the CNN model
class YawnCNN(nn.Module):
    def __init__(self):
        super(YawnCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Load the trained model (on CPU)
model = YawnCNN()
model.load_state_dict(torch.load('yawn_detection.pth', map_location=torch.device('cpu')))
model.eval()

# Transformation to match input format
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Open video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    input_image = Image.fromarray(input_image)
    input_tensor = transform(input_image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = (output > 0.5).float()

    if prediction.item() == 1:
        label = "Yawning"
        color = (0, 0, 255)  # Red for yawning
    else:
        label = "Not Yawning"
        color = (0, 255, 0)  # Green for not yawning

    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


    cv2.imshow('Yawn Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
