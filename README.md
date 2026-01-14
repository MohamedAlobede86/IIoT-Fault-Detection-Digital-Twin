IIoT Digital Twin Diagnostic Framework
Intelligent Fault Detection for Tennessee Eastman Process (TEP)
 Overview
This project presents an integrated framework for real-time fault diagnosis in industrial systems using a Digital Twin approach. It combines a high-performance XGBoost classifier with SHAP (SHapley Additive exPlanations) for model interpretability, all wrapped in a user-friendly JavaFX interface.

 Tech Stack
Backend: Python 3.12 (Flask API).

Machine Learning: XGBoost, Scikit-learn, Joblib.

Explainable AI (XAI): SHAP.

Frontend: JavaFX (JDK 17+).

Communication: RESTful API (HTTP POST).

 Project Structure
app.py: The Flask server hosting the ML model and XAI engine.

model/: Contains pre-trained .pkl files (Classifier, Scaler, Encoder).

lib/: JavaFX dependencies and DLLs for system compatibility.

data/: Sample TEP datasets (CSV) for testing.

Start_System.bat: One-click execution script for Windows.

TEP-GUI.jar: The graphical user interface.

 Quick Start (Installation & Run)
Follow these three steps to run the system:

1. Install Python Dependencies: Open your terminal and run:

Bash

pip install flask xgboost shap joblib matplotlib scikit-learn
2. Download the System: Download the IIoT_Diagnostic_Framework_v1.zip from the Releases section.

3. Launch the Framework: Double-click on Start_System.bat. This will automatically:

Start the Python Backend on http://127.0.0.1:5000.

Launch the JavaFX Graphical Interface.

 System Capabilities
Real-time Prediction: Detects 21+ types of industrial faults.

Explainability: Generates SHAP bar plots for every prediction to show which sensors triggered the alert.

Automated Logging: Saves all diagnostic history into results/prediction_history.csv.