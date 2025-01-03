# Sms_smap_classification_model

# SMS Spam Classification

This project focuses on building a machine learning model to classify SMS messages as spam or not spam. The process involves various stages, including data cleaning, exploratory data analysis (EDA), text preprocessing, model building, evaluation, and further improvements to enhance performance.

---

## Table of Contents
1. [Overview](#overview)
2. [Technologies Used](#technologies-used)
3. [Steps Involved](#steps-involved)
4. [How to Use](#how-to-use)
5. [Results](#results)
6. [Future Improvements](#future-improvements)

---

## Overview
SMS spam classification is a critical task in the field of natural language processing (NLP). This project leverages machine learning to identify and filter spam messages effectively. A dataset of SMS messages was used to train and evaluate the model.

---

## Technologies Used
- Python
- Jupyter Notebook
- Machine Learning Libraries: scikit-learn, pandas, NumPy
- Text Preprocessing Libraries: NLTK
- Streamlit for building a web application

---

## Steps Involved

### 1. Data Cleaning
- Removed unnecessary characters, duplicates, and irrelevant data.
- Ensured the dataset had consistent formatting.

### 2. Exploratory Data Analysis (EDA)
- Visualized data distributions and class imbalances.
- Identified trends and patterns in spam vs. non-spam messages.

### 3. Text Preprocessing
- Tokenized text into individual words.
- Removed stopwords and performed stemming.
- Vectorized text data using TF-IDF for model compatibility.

### 4. Model Building
- Trained multiple models, including Naive Bayes and Logistic Regression.
- Used a pipeline for preprocessing and classification.

### 5. Evaluation
- Evaluated models using metrics such as accuracy, precision, recall, and F1-score.
- Achieved a satisfactory balance between precision and recall.

### 6. Improvement
- Tuned hyperparameters for optimal performance.
- Addressed class imbalance using oversampling and SMOTE techniques.

---

# Results
# Final Model: StackingClassifier
# Base Estimator: Support Vector Machine (SVM)
# Final Estimator: Random Forest Classifier
# Accuracy: 98.16%
# Precision: 96.12%
This StackingClassifier model leverages the strengths of a Support Vector Machine (SVM) as the base estimator and a Random Forest Classifier as the final meta-estimator. The combination achieves outstanding performance, balancing accuracy and precision effectively for SMS spam classification.

## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sms-spam-classifier.git
