import torch
from torchvision import transforms
from PIL import Image
from model import Model

# Load the model
model = Model()
model.load_state_dict(torch.load('model.pt'))
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the image
image_path = 'processed_dataset/1/0.jpg'
image = Image.open(image_path)
input_tensor = transform(image).unsqueeze(0)

# Perform inference
with torch.no_grad():
    output = model(input_tensor)

# Get the predicted class
_, predicted_idx = torch.max(output, 1)
predicted_class = predicted_idx.item()

# Map the predicted class index to its corresponding label
class_labels = ['H', 'E', 'L', 'O']
predicted_label = class_labels[predicted_class]

print(f'Predicted class: {predicted_label}')
