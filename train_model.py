# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load data
df = pd.read_csv("train.csv")
X = df.drop("Cover_Type", axis=1)
y = df["Cover_Type"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train small RandomForest model
model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
dump(model, "forest_cover_model.joblib")
dump(scaler, "scaler.joblib")
