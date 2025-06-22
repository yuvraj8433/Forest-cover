# 🌲 Forest Cover Type Prediction

This project uses machine learning to predict the type of forest cover from geographical and environmental features using a dataset from Roosevelt National Forest, Colorado. The application is deployed using Streamlit for interactive web-based predictions.

## 🚀 Live Demo

🔗 https://forest-cover-yuvraj.streamlit.app/


---

## 📌 Objective

Predict the forest cover type (one of 7 classes) for a 30m x 30m patch of land based on numerical and categorical features using a classification model.

---

## 📂 Dataset

The dataset comes from the **UCI Machine Learning Repository** and contains:

- **10 Numerical Features** (elevation, slope, distances, etc.)
- **4 Wilderness Area Binary Features**
- **41 Soil Type Binary Features**
- **Target**: Cover_Type (1 to 7)

### Cover Types:
1. Spruce/Fir  
2. Lodgepole Pine  
3. Ponderosa Pine  
4. Cottonwood/Willow  
5. Aspen  
6. Douglas-fir  
7. Krummholz  

---

## ⚙️ Features Used

| Feature                              | Description                                     |
|--------------------------------------|-------------------------------------------------|
| Elevation                            | In meters                                      |
| Aspect                               | Azimuth direction in degrees                  |
| Slope                                | Gradient in degrees                           |
| Horizontal & Vertical Distance       | To hydrology (surface water features)         |
| Distance to Roadways & Fire Points   | In meters                                     |
| Hillshade (9am, Noon, 3pm)           | Sunlight index (0–255)                        |
| Wilderness_Area (4 binary columns)   | Wilderness zone (1 = present)                 |
| Soil_Type (41 binary columns)        | Soil types (1 = present)                      |

---

## 🧠 Model

- **Algorithm**: Random Forest Classifier  
- **Libraries**: `scikit-learn`, `joblib`, `streamlit`, `pandas`, `numpy`

---

## 💻 How to Run Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/yuvraj8433/Forest-cover.git
   cd Forest-cover
