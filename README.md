

Explainable Hybrid Machine Learning Framework for Fault Detection in Industrial IoT Systems

Overview

This repository presents an explainable hybrid machine learning framework for fault detection in Industrial Internet of Things (IIoT) environments using the Tennessee Eastman Process (TEP) benchmark dataset.

The framework integrates unsupervised representation learning with supervised classification to improve fault detection reliability in complex industrial systems.

The proposed pipeline combines:

- AutoEncoder for learning compact latent representations of normal system behavior and supporting anomaly detection through reconstruction patterns.
- XGBoost for robust multi-class fault classification.
- SHAP (SHapley Additive exPlanations) to provide feature-level explanations for model predictions.

A lightweight JavaFX graphical interface is provided to demonstrate how the diagnostic framework can be integrated into a monitoring environment.

---

Tech Stack

Backend

- Python 3.12
- Flask API

Machine Learning

- AutoEncoder (Keras / TensorFlow)
- XGBoost
- Scikit-learn
- Joblib

Explainable AI (XAI)

- SHAP (SHapley Additive exPlanations)

Frontend

- JavaFX (JDK 17+)

Research & Development

- Google Colab (Python Notebook)

---

Project Structure

app.py
Flask server hosting the ML inference engine and SHAP explanation module.

Notebook.ipynb
Research notebook containing data preprocessing, feature engineering,
AutoEncoder training, and XGBoost model development.

model/
Contains trained models:
- autoencoder_model.pkl
- xgboost_classifier.pkl
- scaler.pkl
- encoder.pkl

lib/
JavaFX dependencies required for the graphical interface.

data/
Sample Tennessee Eastman Process (TEP) datasets in CSV format.

Start_System.bat
One-click script to launch the backend and graphical interface.

TEP-GUI.jar
JavaFX graphical interface for real-time diagnostic visualization.

---

Quick Start (How to Run)

1. Download the System Package

Navigate to the Releases section of this repository and download:

Stable Release – Full System.zip

Extract the ZIP archive to your local machine.

---

2. Install Python Dependencies

Run the following command:

pip install flask xgboost shap joblib matplotlib scikit-learn tensorflow

---

3. Launch the Framework

Double-click:

Start_System.bat

This will automatically:

1. Start the Python backend (Flask API).
2. Launch the JavaFX monitoring interface.

---

System Capabilities

Hybrid Fault Detection Pipeline

Integrates AutoEncoder-based representation learning with XGBoost classification for robust industrial fault diagnosis.

Multi-class Fault Identification

Detects multiple industrial fault conditions in the Tennessee Eastman Process.

Explainable AI Diagnostics

Each prediction is accompanied by SHAP-based explanations, highlighting the sensor variables that influenced the decision.

Real-Time Monitoring Interface

The JavaFX interface demonstrates how the diagnostic engine can be integrated into a monitoring dashboard.

Automated Logging

All predictions and diagnostic results are stored in:

results/prediction_history.csv

---

Dataset

This project uses the Tennessee Eastman Process (TEP) dataset, a widely used benchmark for industrial fault diagnosis research.

Reference:

Downs, J. J., & Vogel, E. F. (1993).
A plant-wide industrial process control problem.
Computers & Chemical Engineering.

---

Research Context

This repository accompanies the research study:

"Explainable Hybrid Machine Learning Framework for Fault Detection in Industrial IoT Systems."

The work explores how combining representation learning, ensemble machine learning, and explainable AI can improve transparency and reliability in industrial fault diagnosis systems.

---

Author

Mohamed AbdulAli
Industrial AI & IIoT Research