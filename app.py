import streamlit as st
import numpy as np
from joblib import load

st.set_page_config(page_title="Forest Cover Type Predictor", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #f0f2f6; }
    .block-container { padding-top: 1rem; }
    .st-emotion-cache-1kyxreq { padding-bottom: 0rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌲 Forest Cover Type Predictor")
st.markdown("##### Predict the type of forest cover for a 30x30m plot using terrain and vegetation data.")

# Load model and scaler
model = load("forest_cover_model.joblib")
scaler = load("scaler.joblib")

# Input layout
with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        elevation = st.number_input("🏔 Elevation (m)", value=2500)
        slope = st.number_input("⛰ Slope (°)", value=10)
        hd_hydro = st.number_input("💧 Horizontal Distance to Hydrology", value=100)
        hillshade_9am = st.slider("☀️ Hillshade at 9am", 0, 255, 200)

    with col2:
        aspect = st.number_input("🧭 Aspect (°)", value=90)
        vd_hydro = st.number_input("📏 Vertical Distance to Hydrology", value=0)
        hd_road = st.number_input("🛣 Horizontal Distance to Roadways", value=200)
        hillshade_noon = st.slider("🌞 Hillshade at Noon", 0, 255, 220)

    with col3:
        hillshade_3pm = st.slider("🌇 Hillshade at 3pm", 0, 255, 180)
        hd_fire = st.number_input("🔥 Distance to Fire Points", value=500)
        wilderness = st.selectbox("🏞 Wilderness Area", ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"])
        soil_type_index = st.slider("🧱 Soil Type Index", 0, 39, 5)

# One-hot encoding
wilderness_map = {"Rawah": 0, "Neota": 1, "Comanche Peak": 2, "Cache la Poudre": 3}
wilderness_encoded = [1 if i == wilderness_map[wilderness] else 0 for i in range(4)]
soil_encoded = [1 if i == soil_type_index else 0 for i in range(41)]

# Combine all features
input_data = [[
    elevation, aspect, slope, hd_hydro, vd_hydro, hd_road,
    hillshade_9am, hillshade_noon, hillshade_3pm, hd_fire
] + wilderness_encoded + soil_encoded]

# Scale and predict
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

cover_map = {
    1: "🌲 Spruce/Fir", 2: "🌲 Lodgepole Pine", 3: "🌲 Ponderosa Pine",
    4: "🌳 Cottonwood/Willow", 5: "🍂 Aspen", 6: "🌿 Douglas-fir", 7: "🌾 Krummholz"
}

# Display prediction
with st.container():
    st.markdown("---")
    st.subheader("📌 Predicted Forest Cover Type:")
    st.success(f"**{cover_map[prediction]}**", icon="🌳")
