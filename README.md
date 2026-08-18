**Lung-Cancer-Predictive-System**
<br>
<br>
**Project Overview**
<br>
This project applies Artificial Intelligence techniques to assist in the early detection of lung cancer. It combines classical machine learning models trained on structured patient survey data with a Convolutional Neural Network (CNN) trained on histopathological images. A Streamlit GUI integrates both approaches, allowing users to input patient details and upload medical images to generate a transparent diagnostic report.
<br>
<hr>
<br>

__Features__
<br>
-> Implementation of six classical ML models: Logistic Regression, KNN, Decision Tree, Random Forest, Naive Bayes, and SVM.<br>
-> Custom CNN architecture for histopathological image classification.<br>
-> User‑friendly GUI built with Streamlit.<br>
-> Hybrid diagnostic framework combining structured data and image analysis.<br>
-> Transparent reporting with individual model predictions and final integrated decision.
<br>
<hr>
<br>

**Datasets**
<br>
-> Survey Dataset : survey_lung_cancer.csv containing patient records and attributes.<br>
-> Image Dataset : lung_colon_image_set containing histopathological images of lung and colon cancer.
<br>
<hr>
<br>

**Model Archietecture**
<br>
-> Machine Learning models : Classical ML models applied to survey data for binary classification.
<br>
-> CNN Model : Convolutional layers with ReLU activation.
<br>
Max pooling for dimensionality reduction.
<br>
Fully connected dense layers with dropout regularization.
<br>
Final softmax output for classification.
<br>
<hr>
<br> 

**Installation & Usage**
<br>
1. Clone the repository:<br>
git clone https://github.com/Maanvijangir/Lung-Cancer-Predictive-System<br>
cd lung-cancer-Predictive-System
<br>
<br>
2. Install dependencies:
<br>
pip install -r requirements.txt
<br>
<br>
3. Run the Streamlit app:
<br>
streamlit run predict_app.py
<br>
<hr>
<br>

**Results**
<br>
-> Machine Learning models achieved varying accuracy on survey data, with Random Forest and SVM performing strongly.
<br>
-> CNN achieved high accuracy on image classification tasks.
<br>
-> The integrated GUI provides a combined diagnostic report for better reliability.
