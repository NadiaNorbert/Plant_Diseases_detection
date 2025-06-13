from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import io
from torchvision import models
from torch import nn
from fastapi import Query
from weatherapi import get_precaution_weather_report

# type: ignore

app = FastAPI()

# Load the model properly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.mobilenet_v2(weights=None)  # Don't load pretrained to avoid mismatch
model.classifier[1] = nn.Linear(1280, 15)  # Set this to match .pth output
model.load_state_dict(torch.load("model/plant_disease_model.pth", map_location="cpu"))
model.eval()

# Class names
class_names = {
  0:"Pepper__bell___Bacterial_spot",
  1:"Pepper__bell___healthy",
  2:"Potato___Early_blight",
  3:"Potato___Late_blight",
  4:"Potato___healthy",
  5:"Tomato_Bacterial_spot",
  6:"Tomato_Early_blight",
  7:"Tomato_Late_blight",
  8:"Tomato_Leaf_Mold",
  9:"Tomato_Septoria_leaf_spot",
  10:"Tomato_Spider_mites_Two_spotted_spider_mite",
  11:"Tomato__Target_Spot",
  12:"Tomato__Tomato_YellowLeaf__Curl_Virus",
  13:"Tomato__Tomato_mosaic_virus",
  14:"Tomato_healthy"

}
remedies = {
    "Pepper__bell___Bacterial_spot": "Use copper-based bactericides. Avoid overhead watering. Use resistant varieties and crop rotation.",
    "Pepper__bell___healthy": "No remedy needed. Maintain good air circulation and avoid wetting the foliage.",
    "Potato___Early_blight": "Apply fungicides like chlorothalonil or mancozeb early. Rotate crops and remove infected debris.",
    "Potato___Late_blight": "Use fungicides with metalaxyl or chlorothalonil. Destroy infected plants and avoid watering late in the day.",
    "Potato___healthy": "No remedy needed. Ensure proper soil drainage and avoid planting near previously infected areas.",
    "Tomato_Bacterial_spot": "Apply copper-based sprays. Remove infected leaves. Rotate crops yearly.",
    "Tomato_Early_blight": "Use fungicides such as mancozeb or chlorothalonil. Remove lower infected leaves and mulch around the base.",
    "Tomato_Late_blight": "Apply fungicides with chlorothalonil. Destroy infected plants completely. Avoid overhead watering.",
    "Tomato_Leaf_Mold": "Increase airflow. Apply sulfur-based fungicides. Avoid high humidity in greenhouses.",
    "Tomato_Septoria_leaf_spot": "Remove infected leaves. Apply fungicides like copper or mancozeb. Keep foliage dry.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spray neem oil or insecticidal soap. Introduce predatory mites. Avoid dusty conditions.",
    "Tomato__Target_Spot": "Use azoxystrobin or chlorothalonil fungicides. Remove diseased leaves. Avoid overcrowding.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Remove infected plants. Control whiteflies using insecticides. Use resistant tomato varieties.",
    "Tomato__Tomato_mosaic_virus": "Remove and burn infected plants. Disinfect tools. Avoid smoking near plants (virus spreads via tobacco).",
    "Tomato_healthy": "No remedy needed. Keep soil nutrient-rich and water regularly."
}

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    disease_name = class_names[predicted.item()]
    remedy = remedies.get(disease_name, "Remedy information not available.")

    return {
        "disease": disease_name,
        "confidence": f"{confidence.item() * 100:.2f}%",
        "remedy": remedy
}

@app.get("/weather-precaution")
def get_weather_precaution(location: str = Query(..., description="Crop field location")):
    result = get_precaution_weather_report(location)
    return result