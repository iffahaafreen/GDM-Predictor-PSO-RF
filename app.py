import gradio as gr
import numpy as np
import pandas as pd
import joblib
import json

# Load model, scaler, and selected features
model = joblib.load("GDM_PSO_RF_Model.joblib")
scaler = joblib.load("scaler.joblib")
with open("selected_features.json", "r") as f:
    selected_features = json.load(f)

# Full feature list
feature_names = [
    'TCF7L2', 'KCNQ1', 'CDKAL1', 'ADIPOQ', 'FTO', 'CXCL12', 'MEF2C', 'MMP2', 'SOX17',
    'THBS2', 'BMP4', 'Age', 'BMI', 'Pregnancies', 'Glucose', 'BloodPressure',
    'FamilyDiabetes', 'HighBP', 'PhysicalActivity', 'Smoking', 'Stress',
    'Gene_Risk_Score', 'Expression_Risk_Score', 'Clinical_Risk_Score'
]

# Binary features (for radio buttons)
binary_features = ['FamilyDiabetes', 'HighBP', 'PhysicalActivity', 'Smoking', 'Stress']

# Gradio prediction function
def predict_gdm(*inputs):
    input_array = np.array(inputs).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    selected_input = scaled_input[:, selected_features]
    prediction = model.predict(selected_input)[0]
    proba = model.predict_proba(selected_input)[0][1]
    result = "🔴 High Risk of GDM" if prediction == 1 else "🟢 Low Risk of GDM"
    return f"{result} (Probability: {proba:.2%})"

# Gradio input components
inputs = []
for feature in feature_names:
    if feature in binary_features:
        inputs.append(gr.Radio(choices=["No", "Yes"], label=f"{feature} (Yes/No)", type="index"))
    elif feature in ['Age', 'Pregnancies', 'BloodPressure']:
        inputs.append(gr.Number(label=feature, precision=0))
    else:
        inputs.append(gr.Number(label=feature, precision=2))

# Gradio Interface
iface = gr.Interface(
    fn=predict_gdm,
    inputs=inputs,
    outputs="text",
    title="GDM Predictor (PSO-RF)",
    description="Enter 24 clinical and genetic features to assess GDM risk."
)

iface.launch()
