# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oGLun-6oklmpl5SIDhmBmVzuGQ4jCjwE
"""

!pip install streamlit
!pip install torch torchvision

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)          # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy} # Combine accuracies

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))

import torch
from torchvision import transforms
from PIL import Image
import streamlit as st

# Define the class names for plant diseases
classes = ['Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Potato___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Strawberry___Leaf_scorch', 'Peach___healthy', 'Apple___Apple_scab', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot', 'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 'Pepper,_bell___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus', 'Strawberry___healthy', 'Apple___healthy', 'Grape___Black_rot', 'Potato___Early_blight', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)', 'Raspberry___healthy', 'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Pepper,_bell___Bacterial_spot', 'Corn_(maize)___healthy']

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the saved model as an OrderedDict
checkpoint = torch.load('plant-disease-model (2).pth', map_location=torch.device('cpu'))

# Check if the checkpoint is an OrderedDict
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    # Extract the state dictionary from the checkpoint
    state_dict = checkpoint['state_dict']
else:
    # The checkpoint is likely the state dictionary itself
    state_dict = checkpoint

# Create an instance of your model
# (Replace 'YourModelClass' with the actual class name of your model)
model = ImageClassificationBase()

# Load the state dictionary into the model
# Assuming 'model' is an instance of ImageClassificationBase
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
model.load_state_dict(filtered_state_dict)


# Set the model to evaluation mode
model.eval()

def classify_image(image):
    # Apply the transformation to the image
    image_tensor = transform(image).unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        class_index = predicted.item()

    return class_index

st.title("Plant Disease Classification")

image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if image_file is not None:
    image = Image.open(image_file)
    st.image(image, width=250)
    class_index = classify_image(image)
    class_name = classes[class_index]
    st.header(class_name)

