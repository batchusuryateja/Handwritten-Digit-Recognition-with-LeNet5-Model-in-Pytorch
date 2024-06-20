import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image
import numpy as np
import io

# Defining model classes (MLP, SimpleCNN, LeNet5)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiating and loading the models
mlp_model = MLP()
cnn_model = SimpleCNN()
lenet_model = LeNet5()

mlp_model.load_state_dict(torch.load('MLP_model.pth', map_location=torch.device('cpu')))
cnn_model.load_state_dict(torch.load('CNN_model.pth', map_location=torch.device('cpu')))
lenet_model.load_state_dict(torch.load('LeNet5_model.pth', map_location=torch.device('cpu')))

mlp_model.eval()
cnn_model.eval()
lenet_model.eval()

# Defining the transformation to preprocess the image during inference
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_image(model, image):
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Streamlit app
st.title("Hand Written Digit Recognition")
st.write("Upload an image of a digit and get predictions from one of three models: MLP, SimpleCNN, or LeNet5, or all.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Model selection dropdown menu
model_name = st.selectbox("Select model for prediction:", ("MLP", "SimpleCNN", "LeNet5", "All"))

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert('L')
    resized_image = image.resize((150, 150))  # Resize the image for display
    st.image(resized_image, caption='Uploaded Image.', use_column_width=False)
    st.write("")
    st.write("Classifying...")

    # Predicting using the selected model
    if model_name == "MLP":
        prediction = predict_image(mlp_model, image)
        st.write(f"MLP Prediction: {prediction}")
    elif model_name == "SimpleCNN":
        prediction = predict_image(cnn_model, image)
        st.write(f"SimpleCNN Prediction: {prediction}")
    elif model_name == "LeNet5":
        prediction = predict_image(lenet_model, image)
        st.write(f"LeNet5 Prediction: {prediction}")
    else:
        mlp_prediction = predict_image(mlp_model, image)
        cnn_prediction = predict_image(cnn_model, image)
        lenet_prediction = predict_image(lenet_model, image)
        st.write(f"MLP Prediction: {mlp_prediction}")
        st.write(f"SimpleCNN Prediction: {cnn_prediction}")
        st.write(f"LeNet5 Prediction: {lenet_prediction}")