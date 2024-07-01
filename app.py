# Import necessary modules from Flask
from flask import Flask, request, render_template, redirect, url_for, flash
 
# Import PIL for image processing
from PIL import Image

# Import PyTorch and related libraries for neural networks and image transformations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Import os for file handling, base64 for encoding/decoding images, and BytesIO for in-memory file handling
import os
import base64
from io import BytesIO

# Initialize Flask app and set a secret key for session management
app = Flask(__name__)
app.secret_key = 'secret_key'

# Define the models 
class MLP(nn.Module):
    def __init__(self):
    # Initialize the MLP model with three fully connected layers
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  #input layer
        self.fc2 = nn.Linear(128, 64)  #Hidden layer
        self.fc3 = nn.Linear(64, 10)   #output layer

    def forward(self, x):
    # Define the forward pass for the MLP model
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
    # Initialize the SimpleCNN model with convolutional and fully connected layers
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
    # Define the forward pass for the SimpleCNN model
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LeNet5(nn.Module):
    def __init__(self):
    # Initialize the LeNet5 model with convolutional and fully connected layers
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
    # Define the forward pass for the LeNet5 model
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the models
# Initialize and load the pre-trained MLP model
mlp_model = MLP()
mlp_model.load_state_dict(torch.load('MLP_model.pth'))
mlp_model.eval()

# Initialize and load the pre-trained SimpleCNN model
cnn_model = SimpleCNN()
cnn_model.load_state_dict(torch.load('CNN_model.pth'))
cnn_model.eval()

# Initialize and load the pre-trained LeNet5 model
lenet_model = LeNet5()
lenet_model.load_state_dict(torch.load('LeNet5_model.pth'))
lenet_model.eval()

# Define a series of transformations to preprocess the input image
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to load and preprocess an image from a given path
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to evaluate a single image using a specified model
def evaluate_single_image(model, image_path):
    image = load_and_preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        predicted_label = predicted.item()
    return predicted_label

# Function to preprocess an image drawn on a canvas (base64 encoded)
def preprocess_canvas_image(image_data):
    image = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1])))
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Define the route for the index page, handling file upload and model prediction
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file:
            model_choice = request.form.get('model_choice')

            # Create the 'static/uploads' directory if it doesn't exist
            upload_folder = os.path.join('static', 'uploads')
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            image_path = os.path.join(upload_folder, file.filename)
            file.save(image_path)

            if model_choice == 'mlp':
                predicted_label = evaluate_single_image(mlp_model, image_path)
            elif model_choice == 'cnn':
                predicted_label = evaluate_single_image(cnn_model, image_path)
            elif model_choice == 'lenet':
                predicted_label = evaluate_single_image(lenet_model, image_path)
            else:
                flash('Invalid model choice', 'error')
                return redirect(request.url)

            return render_template('result.html', prediction=predicted_label, image_path=image_path)

    return render_template('index.html')

# Define the route to handle predictions from canvas drawings
@app.route('/predict_canvas', methods=['POST'])
def predict_canvas():
    model_choice = request.form.get('model_choice_canvas')
    canvas_data = request.form.get('canvasData')

    if not canvas_data:
        flash('No drawing data', 'error')
        return redirect(url_for('index'))

    image = preprocess_canvas_image(canvas_data)

    if model_choice == 'mlp':
        model = mlp_model
    elif model_choice == 'cnn':
        model = cnn_model
    elif model_choice == 'lenet':
        model = lenet_model
    else:
        flash('Invalid model choice', 'error')
        return redirect(url_for('index'))

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        predicted_label = predicted.item()

    return render_template('result.html', prediction=predicted_label, image_path=None)

# Run the Flask application in debug mode
if __name__ == '__main__':
    app.run(debug=True)
