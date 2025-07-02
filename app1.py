import streamlit as st
import numpy as np
import pandas as pd
import pickle
import config
from utils.fertilizer import fertilizer_dic
from utils.disease import disease_dic
import requests
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
import io

# Load models
disease_classes = [...]  # same list from your Flask app

disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load('models/plant_disease_model.pth', map_location=torch.device('cpu')))
disease_model.eval()

crop_model = pickle.load(open('models/RandomForest.pkl', 'rb'))

# Weather function
def weather_fetch(city):
    api_key = config.weather_api_key
    url = f"http://api.openweathermap.org/data/2.5/weather?appid={api_key}&q={city}"
    response = requests.get(url).json()
    if response["cod"] != "404":
        temp = round((response["main"]["temp"] - 273.15), 2)
        humidity = response["main"]["humidity"]
        return temp, humidity
    else:
        return None

# Disease prediction
def predict_image(img_bytes):
    transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    image = Image.open(io.BytesIO(img_bytes))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    yb = disease_model(img_u)
    _, preds = torch.max(yb, dim=1)
    return disease_classes[preds[0].item()]

# Streamlit UI
st.set_page_config(page_title="Harvestify", layout="centered")
st.title("🌾 Harvestify - Smart Farming Assistant")

menu = st.sidebar.selectbox("Choose Feature", ["Home", "Crop Recommendation", "Fertilizer Suggestion", "Disease Detection"])

if menu == "Home":
    st.image("https://images.unsplash.com/photo-1602468635045-dc9141f26d65", use_column_width=True)
    st.markdown("Welcome to **Harvestify** - Your AI-powered Agri Assistant.")

elif menu == "Crop Recommendation":
    st.subheader("🌱 Crop Recommendation")
    N = st.number_input("Nitrogen", min_value=0)
    P = st.number_input("Phosphorous", min_value=0)
    K = st.number_input("Potassium", min_value=0)
    ph = st.number_input("pH", min_value=0.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
    city = st.text_input("Enter City Name")

    if st.button("Predict Crop"):
        weather = weather_fetch(city)
        if weather:
            temp, humidity = weather
            data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
            prediction = crop_model.predict(data)[0]
            st.success(f"🌾 Recommended Crop: **{prediction}**")
        else:
            st.error("City not found. Try again.")

