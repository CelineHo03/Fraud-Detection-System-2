#!/bin/bash
# Activate fraud detection virtual environment

source fraud_detection_env/bin/activate

echo "🐍 Fraud Detection Environment Activated (venv)"
echo "Python: $(which python)"
echo "Virtual env: $VIRTUAL_ENV"
