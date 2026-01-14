IIoT Digital Twin Diagnostic Framework
Intelligent Fault Detection for Tennessee Eastman Process (TEP)
Overview
This project presents an integrated framework for real-time fault diagnosis in industrial systems using a Digital Twin approach. It combines a high-performance XGBoost classifier with SHAP (SHapley Additive exPlanations) for model interpretability, all wrapped in a user-friendly JavaFX interface.

 Tech Stack
Backend: Python 3.12 (Flask API).

Machine Learning: XGBoost, Scikit-learn, Joblib.

Explainable AI (XAI): SHAP (SHapley Additive exPlanations).

Frontend: JavaFX (JDK 17+).

Research & Development: Google Colab (Python Notebook).

 Project Structure
app.py: The Flask server hosting the ML model and XAI engine.

Notebook.ipynb: The full research notebook containing data cleaning, preprocessing, and model training phases.

model/: Contains pre-trained .pkl files (Classifier, Scaler, Encoder).

lib/: JavaFX dependencies and DLLs for system compatibility.

data/: Sample TEP datasets (CSV) for testing.

Start_System.bat: One-click execution script for Windows.

TEP-GUI.jar: The graphical user interface.

 Quick Start (How to Run)
To experience the full system, please follow these steps:

1. Download the Full Package:

Navigate to the Releases section on the right sidebar.

Download the Stable Release - Full System.zip file.

Extract the ZIP file to your local machine.

2. Install Python Dependencies:

Bash

pip install flask xgboost shap joblib matplotlib scikit-learn
3. Launch the Framework:

Double-click on Start_System.bat.

This will automatically start the Python backend and launch the JavaFX GUI.

 System Capabilities
End-to-End Pipeline: From raw TEP data cleaning to real-time industrial fault detection.

Real-time Prediction: Detects 21+ types of industrial faults with high accuracy.

Explainability (XAI): Provides SHAP bar plots for every prediction to explain the impact of each sensor on the diagnostic result.

Automated Logging: Saves all diagnostic history into results/prediction_history.csv.
