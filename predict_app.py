import streamlit as st
import pandas as pd
import joblib
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# -------------------------------
# Load classical ML models
# -------------------------------
log_reg = joblib.load("log_reg.joblib")
knn = joblib.load("knn.joblib")
d_tree = joblib.load("decision_tree.joblib")
svm = joblib.load("svm.joblib")
nb = joblib.load("naive_bayes.joblib")
rb = joblib.load("random_forest.joblib")

models = {
    "Logistic Regression": log_reg,
    "KNN": knn,
    "Decision Tree": d_tree,
    "SVM": svm,
    "Naive Bayes": nb,
    "Random Forest": rb,
}

# Load metrics
with open("model_metrics.json", "r") as f:
    metrics = json.load(f)

# -------------------------------
# Define CNN architecture (must match training)
# -------------------------------
class LungCancerCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(LungCancerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------------------
# Load CNN model
# -------------------------------
cnn_model = torch.load("cancer_cnn_model3.pth", map_location="cpu", weights_only=False)
cnn_model.eval()
cnn_labels = ["No Cancer", "Cancer"]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🫁 Lung Cancer Diagnostic App")
st.write("Enter patient details and upload a histopathology image to generate a combined diagnostic report.")

# -------------------------------
# Patient Data Input
# -------------------------------
st.subheader("Patient Data Input")
gender = st.selectbox("Gender", ["Male", "Female"], index=0)
age = st.slider("Age", 20, 100, 65)
smoking = st.selectbox("Smoking", ["No", "Yes"], index=1)
yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"], index=1)
anxiety = st.selectbox("Anxiety", ["No", "Yes"], index=1)
peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"], index=1)
chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"], index=1)
fatigue = st.selectbox("Fatigue", ["No", "Yes"], index=1)
allergy = st.selectbox("Allergy", ["No", "Yes"], index=0)
wheezing = st.selectbox("Wheezing", ["No", "Yes"], index=1)
alcohol = st.selectbox("Alcohol Consuming", ["No", "Yes"], index=1)
coughing = st.selectbox("Coughing", ["No", "Yes"], index=1)
shortness_of_breath = st.selectbox("Shortness of Breath", ["No", "Yes"], index=1)
swallowing_difficulty = st.selectbox("Swallowing Difficulty", ["No", "Yes"], index=1)
chest_pain = st.selectbox("Chest Pain", ["No", "Yes"], index=1)

gender_val = 0 if gender == "Male" else 1
smoking_val = 1 if smoking == "Yes" else 0
yellow_fingers_val = 1 if yellow_fingers == "Yes" else 0
anxiety_val = 1 if anxiety == "Yes" else 0
peer_pressure_val = 1 if peer_pressure == "Yes" else 0
chronic_disease_val = 1 if chronic_disease == "Yes" else 0
fatigue_val = 1 if fatigue == "Yes" else 0
allergy_val = 1 if allergy == "Yes" else 0
wheezing_val = 1 if wheezing == "Yes" else 0
alcohol_val = 1 if alcohol == "Yes" else 0
coughing_val = 1 if coughing == "Yes" else 0
shortness_val = 1 if shortness_of_breath == "Yes" else 0
swallowing_val = 1 if swallowing_difficulty == "Yes" else 0
chest_pain_val = 1 if chest_pain == "Yes" else 0

patient_features = [[gender_val, age, smoking_val, yellow_fingers_val, anxiety_val, peer_pressure_val,
                 chronic_disease_val, fatigue_val, allergy_val, wheezing_val, alcohol_val, coughing_val,
                 shortness_val, swallowing_val, chest_pain_val]]

# -------------------------------
# Image Upload
# -------------------------------
st.subheader("Upload Histopathology Image")
uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

# -------------------------------
# Predict Button
# -------------------------------
if st.button("Predict Cancer Risk"):
    st.subheader("Model Accuracy")
    st.dataframe(pd.DataFrame(metrics.items(), columns=["Model", "Accuracy"]))

    # ML Prediction
    predictions = {}
    for name, model in models.items():
        pred = model.predict(patient_features)[0]
        predictions[name] = "Cancer" if pred == 1 else "No Cancer"
    ml_final = max(set(predictions.values()), key=list(predictions.values()).count)

    st.subheader("Predictions by ML Models")
    st.write(pd.DataFrame(predictions.items(), columns=["Model", "Prediction"]))
    st.write(f"**Final ML Result:** {ml_final}")

    # CNN Prediction
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor()
        ])
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = cnn_model(img_tensor)
            _, pred = torch.max(outputs, 1)
            cnn_result = cnn_labels[pred.item()]

        st.image(uploaded_file, caption="Uploaded Image", width=400)
        st.write(f"**CNN Result:** {cnn_result}")
    else:
        cnn_result = "No Image Provided"
        st.warning("No image uploaded. CNN prediction skipped.")

    # -------------------------------
    # Final Combined Report
    # -------------------------------
    st.subheader("Final Diagnostic Report")
    if ml_final == "Cancer" or cnn_result == "Cancer":
        st.error("Result: Cancer detected")
    else:
        st.success("Result: No Cancer detected")
