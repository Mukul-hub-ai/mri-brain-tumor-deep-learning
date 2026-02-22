import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class Names 
classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Load Model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load("best_resnet50.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Transform 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# UI
st.title(" Brain Tumor MRI Classification")
st.write("Upload an MRI image to predict tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item() * 100

    st.success(f"Predicted Tumor Type: {predicted_class}")
    st.info(f"Confidence: {confidence_score:.2f}%")

    st.subheader("Class Probabilities:")
    for i, cls in enumerate(classes):
        st.write(f"{cls}: {probabilities[0][i].item()*100:.2f}%")
