# Lung-Cancer-Predictive-System

**Project Overview**
<br>
This project applies Artificial Intelligence techniques to assist in the early detection of lung cancer. It combines classical machine learning models trained on structured patient survey data with a Convolutional Neural Network (CNN) trained on histopathological images. A Streamlit GUI integrates both approaches, allowing users to input patient details and upload medical images to generate a transparent diagnostic report.
<hr>

__Features__
- Implementation of six classical ML models: Logistic Regression, KNN, Decision Tree, Random Forest, Naive Bayes, and SVM.<br>
- Custom CNN architecture for histopathological image classification.<br>
- User‑friendly GUI built with Streamlit.<br>
- Hybrid diagnostic framework combining structured data and image analysis.<br>
- Transparent reporting with individual model predictions and final integrated decision.

<hr>

**Datasets**
- Survey Dataset : survey_lung_cancer.csv containing patient records and attributes.<br>
- Image Dataset : lung_colon_image_set containing histopathological images of lung and colon cancer.
<hr>

**Model Archietecture**
- Machine Learning models : Classical ML models applied to survey data for binary classification.
- CNN Model : ->Convolutional layers with ReLU activation.
 ->Max pooling for dimensionality reduction.
 ->Fully connected dense layers with dropout regularization. 
 ->Final softmax output for classification.

<hr>

**Installation & Usage**
1. Clone the repo:
   ```bash
   git clone https://github.com/Maanvijangir/Lung-Cancer-Predictive-System.git
   cd lung-cancer-prediction
2. Installing Dependencies
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit app:
   ```bash
   streamlit run predict_app.py

<hr>

**Results**
- Machine Learning models achieved varying accuracy on survey data, with Random Forest and Logistic regression performing strongly.
- CNN achieved high accuracy on image classification tasks.
- The integrated GUI provides a combined diagnostic report for better reliability.
