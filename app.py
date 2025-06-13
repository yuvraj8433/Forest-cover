# app.py

import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

st.title("🌲 Forest Cover Type Predictor")
st.write("Predict the type of forest cover for a 30x30m plot using terrain and vegetation data.")

# Load model and scaler
model = load("forest_cover_model.joblib")
scaler = load("scaler.joblib")

# UI for input
st.subheader("Enter Terrain Data:")
elevation = st.number_input("Elevation (m)", value=2500)
aspect = st.number_input("Aspect (degrees)", value=90)
slope = st.number_input("Slope (degrees)", value=10)
hd_hydro = st.number_input("Horizontal Distance to Hydrology", value=100)
vd_hydro = st.number_input("Vertical Distance to Hydrology", value=0)
hd_road = st.number_input("Horizontal Distance to Roadways", value=200)
hillshade_9am = st.slider("Hillshade 9am", 0, 255, 200)
hillshade_noon = st.slider("Hillshade Noon", 0, 255, 220)
hillshade_3pm = st.slider("Hillshade 3pm", 0, 255, 180)
hd_fire = st.number_input("Horizontal Distance to Fire Points", value=500)

# Wilderness and Soil - example with basic selection
wilderness = st.selectbox("Wilderness Area", ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"])
soil_type_index = st.slider("Soil Type Index (0 to 39)", 0, 39, 5)

# One-hot encode wilderness and soil
wilderness_map = {"Rawah": 0, "Neota": 1, "Comanche Peak": 2, "Cache la Poudre": 3}
wilderness_encoded = [1 if i == wilderness_map[wilderness] else 0 for i in range(4)]
soil_encoded = [1 if i == soil_type_index else 0 for i in range(41)]


# Combine all features
input_data = [[
    elevation, aspect, slope, hd_hydro, vd_hydro, hd_road,
    hillshade_9am, hillshade_noon, hillshade_3pm, hd_fire
] + wilderness_encoded + soil_encoded]

# Scale and Predict
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

cover_map = {
    1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine",
    4: "Cottonwood/Willow", 5: "Aspen", 6: "Douglas-fir", 7: "Krummholz"
}

st.subheader("🌳 Predicted Forest Cover Type:")
st.success(cover_map[prediction])
