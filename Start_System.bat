@echo off
title IIoT Diagnostic Framework Control Center
color 0B
echo ======================================================
echo    Digital Twin Diagnostic Framework System
echo ======================================================

:: التأكد من الوقوف في المجلد الحالي
cd /d "%~dp0"

echo [1/2] Launching Python Backend...
:: تم حذف كلمة backend/ لأن الملف موجود أمامك مباشرة
start "PYTHON_BACKEND" cmd /k "python app.py"

echo [WAIT] Waiting 5 seconds for initialization...
timeout /t 5

echo [2/2] Launching JavaFX Interface...
:: التعديل ليتناسب مع مجلد lib ومجلد الموديل
java -Djava.library.path="lib" -Dprism.order=sw --module-path "lib" --add-modules javafx.controls,javafx.fxml -jar TEP-GUI.jar

echo ======================================================
echo System is active.
pause