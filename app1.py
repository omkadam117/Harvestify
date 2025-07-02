import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
import config
import torch
from torchvision import transforms
from PIL import Image
import io
from utils.fertilizer import fertilizer_dic
from utils.disease import disease_dic
from utils.model import ResNet9

# -------------------- Load Models -----------------------
disease_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                   'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                   'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                   'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                   'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Load Disease Model
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load('models/plant_disease_model.pth', map_location=torch.device('cpu')))
disease_model.eval()

# Load Crop Recommendation Model
crop_model = pickle.load(open('models/RandomForest.pkl', 'rb'))

# -------------------- Helper Functions -----------------------

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

def predict_image(img_bytes):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    yb = disease_model(img_u)
    _, preds = torch.max(yb, dim=1)
    return disease_classes[preds[0].item()]

# -------------------- Streamlit UI -----------------------

st.set_page_config(page_title="Harvestify", layout="centered")
st.title("🌾 Harvestify - Smart Farming Assistant")

menu = st.sidebar.radio("Choose Feature", ["🏠 Home", "🌱 Crop Recommendation", "🧪 Fertilizer Suggestion", "🦠 Disease Detection"])

if menu == "🏠 Home":
    st.image("https://images.unsplash.com/photo-1602468635045-dc9141f26d65", use_column_width=True)
    st.markdown("Welcome to **Harvestify** - Your AI-powered agriculture assistant for smart crop decisions.")

elif menu == "🌱 Crop Recommendation":
    st.subheader("Crop Recommendation")
    N = st.number_input("Nitrogen (N)", min_value=0)
    P = st.number_input("Phosphorous (P)", min_value=0)
    K = st.number_input("Potassium (K)", min_value=0)
    ph = st.number_input("pH Value", min_value=0.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
    city = st.text_input("City for Weather Info")

    if st.button("Recommend Crop"):
        weather = weather_fetch(city)
        if weather:
            temp, humidity = weather
            data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
            result = crop_model.predict(data)[0]
            st.success(f"🌾 Recommended Crop: **{result}**")
        else:
            st.error("City not found. Please check spelling or try another city.")

elif menu == "🧪 Fertilizer Suggestion":
    st.subheader("Fertilizer Suggestion")
    crop_name = st.text_input("Crop Name (e.g., rice, maize, sugarcane)")
    N = st.number_input("Current Nitrogen (N)", min_value=0)
    P = st.number_input("Current Phosphorous (P)", min_value=0)
    K = st.number_input("Current Potassium (K)", min_value=0)

    if st.button("Suggest Fertilizer"):
        try:
            df = pd.read_csv("Data/fertilizer.csv")
            nr = df[df['Crop'] == crop_name]['N'].iloc[0]
            pr = df[df['Crop'] == crop_name]['P'].iloc[0]
            kr = df[df['Crop'] == crop_name]['K'].iloc[0]

            n = nr - N
            p = pr - P
            k = kr - K
            temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
            max_val = temp[max(temp.keys())]

            if max_val == "N":
                key = "NHigh" if n < 0 else "Nlow"
            elif max_val == "P":
                key = "PHigh" if p < 0 else "Plow"
            else:
                key = "KHigh" if k < 0 else "Klow"

            recommendation = fertilizer_dic[key]
            st.info(recommendation)
        except:
            st.error("Invalid crop name or data missing.")

elif menu == "🦠 Disease Detection":
    st.subheader("Plant Disease Detection")
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        try:
            label = predict_image(bytes_data)
            st.image(uploaded_file, caption="Uploaded Leaf", use_column_width=True)
            st.success(f"Prediction: **{label}**")
            st.info(disease_dic[label])
        except:
            st.error("Prediction failed. Please try with a clearer image.")
