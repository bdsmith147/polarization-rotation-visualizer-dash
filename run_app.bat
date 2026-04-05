@echo off

REM ---- Step 1: Initialize Conda ----
CALL "C:\Users\bensm\anaconda3\Scripts\activate.bat"

REM ---- Step 2: Activate your environment ----
CALL conda activate polrot

REM ---- Step 3: Run Dash app in background ----
START "" cmd /k python app.py

REM ---- Step 4: Wait a moment for server to start ----
timeout /t 3 > nul

REM ---- Step 5: Open browser ----
start http://127.0.0.1:8050