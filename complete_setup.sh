#!/bin/bash

# Complete Setup Script for Fraud Detection System (Fixed for python3)
# This script sets up everything: virtual environment, dependencies, and tests

echo "🚀 Fraud Detection System - Complete Setup (Python3)"
echo "===================================================="

# Function to find the correct Python command
find_python() {
    if command -v python3 &> /dev/null; then
        echo "python3"
    elif command -v python &> /dev/null; then
        # Check if it's Python 3
        python_version=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1)
        if [ "$python_version" = "3" ]; then
            echo "python"
        else
            echo ""
        fi
    else
        echo ""
    fi
}

# Find Python command
PYTHON_CMD=$(find_python)

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Python 3 not found. Please install Python 3."
    echo "   Visit: https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Found Python: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Check if we're in the right directory
if [[ ! -f "requirements.txt" ]] && [[ ! -f "requirements_minimal.txt" ]]; then
    echo "⚠️  requirements.txt not found. Creating a minimal version..."
    
    # Create minimal requirements if missing
    cat > requirements_minimal.txt << 'EOF'
# Minimal Requirements for Testing
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0
shap>=0.43.0
plotly>=5.17.0
streamlit>=1.28.0
pyyaml>=6.0
python-dotenv>=1.0.0
tqdm>=4.66.0
pydantic>=2.4.0
EOF
    echo "✅ Created requirements_minimal.txt"
fi

# Function to check if conda is available
check_conda() {
    if command -v conda &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to setup with conda
setup_conda() {
    echo "🐍 Setting up with Conda"
    
    # Create environment
    echo "Creating conda environment 'fraud_detection'..."
    conda create -n fraud_detection python=3.11 -y
    
    # Activate environment
    echo "Activating conda environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate fraud_detection
    
    # Install core packages via conda
    echo "Installing core packages via conda-forge..."
    conda config --add channels conda-forge
    conda config --set channel_priority strict
    
    conda install -y \
        pandas \
        numpy \
        scikit-learn \
        xgboost \
        plotly \
        streamlit \
        pyyaml \
        tqdm \
        pydantic \
        joblib
    
    # Try to install geospatial packages
    echo "Installing geospatial packages..."
    conda install -y geopandas folium geopy || echo "⚠️  Some geospatial packages may need manual installation"
    
    # Install remaining with pip
    echo "Installing remaining packages with pip..."
    pip install shap python-dotenv streamlit-folium || echo "⚠️  Some packages may need manual installation"
    
    return 0
}

# Function to setup with venv
setup_venv() {
    echo "🐍 Setting up with Python venv"
    
    # Create virtual environment
    echo "Creating virtual environment 'fraud_detection_env'..."
    $PYTHON_CMD -m venv fraud_detection_env
    
    # Activate virtual environment
    echo "Activating virtual environment..."
    source fraud_detection_env/bin/activate
    
    # Verify Python in venv
    echo "Python in venv: $(which python) ($(python --version))"
    
    # Upgrade pip
    echo "Upgrading pip..."
    python -m pip install --upgrade pip
    
    # Install core packages first
    echo "Installing core ML packages..."
    pip install pandas numpy scikit-learn xgboost joblib
    
    echo "Installing visualization packages..."
    pip install plotly seaborn matplotlib
    
    echo "Installing streamlit..."
    pip install streamlit
    
    echo "Installing utilities..."
    pip install pyyaml python-dotenv tqdm pydantic
    
    echo "Installing SHAP..."
    pip install shap
    
    # Try geospatial packages (may fail, that's ok for now)
    echo "Attempting to install geospatial packages..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Installing system dependencies with Homebrew..."
        if command -v brew &> /dev/null; then
            brew install proj geos gdal spatialindex
            echo "Installing geospatial Python packages..."
            pip install geopandas geopy folium streamlit-folium || echo "⚠️  Geospatial packages failed - can be installed later"
        else
            echo "⚠️  Homebrew not found. Skipping geospatial packages for now."
        fi
    else
        echo "⚠️  Non-macOS system. Skipping geospatial packages for now."
    fi
    
    return 0
}

# Function to test core installation
test_core_installation() {
    echo ""
    echo "🧪 Testing core ML packages..."
    
    # Test core ML packages with proper Python command
    python -c "
import sys
print('Testing core packages...')
try:
    import pandas
    print('✅ pandas')
except ImportError as e:
    print(f'❌ pandas: {e}')
    sys.exit(1)

try:
    import numpy
    print('✅ numpy')
except ImportError as e:
    print(f'❌ numpy: {e}')
    sys.exit(1)

try:
    import sklearn
    print('✅ scikit-learn')
except ImportError as e:
    print(f'❌ scikit-learn: {e}')
    sys.exit(1)

try:
    import xgboost
    print('✅ xgboost')
except ImportError as e:
    print(f'❌ xgboost: {e}')
    sys.exit(1)

try:
    import shap
    print('✅ shap')
except ImportError as e:
    print(f'❌ shap: {e}')
    sys.exit(1)

try:
    import streamlit
    print('✅ streamlit')
except ImportError as e:
    print(f'❌ streamlit: {e}')
    sys.exit(1)

print('🎉 Core ML packages working!')
"
    
    return $?
}

# Function to test geographic packages (optional)
test_geo_packages() {
    echo ""
    echo "🗺️ Testing geographic packages (optional)..."
    
    python -c "
try:
    import geopandas
    print('✅ geopandas')
    geo_success = True
except ImportError:
    print('⚠️  geopandas not installed (can be added later)')
    geo_success = False

try:
    import folium
    print('✅ folium')
except ImportError:
    print('⚠️  folium not installed (can be added later)')

if geo_success:
    print('🗺️ Geographic packages available!')
else:
    print('⚠️  Geographic packages not installed - system will work without maps')
"
    
    # Don't fail on geographic packages
    return 0
}

# Function to create activation script
create_activation_script() {
    if check_conda && conda env list | grep -q "fraud_detection"; then
        # Create conda activation script
        cat > activate_fraud_env.sh << 'EOF'
#!/bin/bash
# Activate fraud detection conda environment

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fraud_detection

echo "🐍 Fraud Detection Environment Activated (Conda)"
echo "Python: $(which python)"
echo "Environment: $CONDA_DEFAULT_ENV"
EOF
    else
        # Create venv activation script
        cat > activate_fraud_env.sh << 'EOF'
#!/bin/bash
# Activate fraud detection virtual environment

source fraud_detection_env/bin/activate

echo "🐍 Fraud Detection Environment Activated (venv)"
echo "Python: $(which python)"
echo "Virtual env: $VIRTUAL_ENV"
EOF
    fi
    
    chmod +x activate_fraud_env.sh
    echo "✅ Created activation script: activate_fraud_env.sh"
}

# Function to create test script if missing
create_test_script() {
    if [[ ! -f "test_core.py" ]]; then
        cat > test_core.py << 'EOF'
#!/usr/bin/env python3
"""Test core ML functionality without geographic dependencies."""

import sys

def test_core_packages():
    """Test core ML packages."""
    print("🧪 Testing Core ML Packages")
    print("=" * 30)
    
    packages = ['pandas', 'numpy', 'sklearn', 'xgboost', 'shap', 'streamlit']
    failed = []
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed.append(package)
    
    return len(failed) == 0

def test_basic_functionality():
    """Test basic ML functionality."""
    print("\n🔧 Testing Basic Functionality")
    print("=" * 30)
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        import xgboost as xgb
        
        # Create sample data
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Test train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print("✅ train_test_split")
        
        # Test XGBoost
        model = xgb.XGBClassifier(n_estimators=10)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("✅ XGBoost training and prediction")
        
        # Test pandas
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        print("✅ Pandas DataFrame creation")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Core ML Pipeline Test")
    print("=" * 40)
    
    imports_ok = test_core_packages()
    
    if imports_ok:
        functionality_ok = test_basic_functionality()
        if functionality_ok:
            print("\n🎉 SUCCESS! Core ML pipeline is working!")
            print("You can now proceed with building the fraud detection system.")
        else:
            print("\n❌ Functionality tests failed.")
            sys.exit(1)
    else:
        print("\n❌ Import tests failed.")
        sys.exit(1)
EOF
        chmod +x test_core.py
        echo "✅ Created test_core.py"
    fi
}

# Main setup logic
main() {
    echo "Checking system..."
    
    # Ask user for preference if both are available
    if check_conda; then
        echo ""
        echo "Both conda and venv are available. Which would you prefer?"
        echo "1) Conda (recommended for complex dependencies)"
        echo "2) Python venv (lighter weight)"
        read -p "Enter choice (1 or 2, default=2): " choice
        
        if [[ "$choice" == "1" ]]; then
            setup_method="conda"
        else
            setup_method="venv"
        fi
    else
        echo "Using Python venv..."
        setup_method="venv"
    fi
    
    # Run setup
    if [[ "$setup_method" == "conda" ]]; then
        setup_conda
    else
        setup_venv
    fi
    
    if [ $? -ne 0 ]; then
        echo "❌ Setup failed"
        exit 1
    fi
    
    # Test core installation
    test_core_installation
    if [ $? -ne 0 ]; then
        echo "❌ Core package tests failed"
        exit 1
    fi
    
    # Test geographic packages (optional)
    test_geo_packages
    
    # Create test script
    create_test_script
    
    # Create activation script
    create_activation_script
    
    echo ""
    echo "🎉 SUCCESS! Core setup complete!"
    echo ""
    echo "📋 What was installed:"
    echo "   ✅ Virtual environment created"
    echo "   ✅ Core ML packages (pandas, numpy, sklearn, xgboost, shap)"
    echo "   ✅ Streamlit and visualization packages"
    echo "   ✅ Basic functionality tested"
    echo ""
    echo "🗺️ Geographic packages:"
    echo "   ⚠️  May need separate installation (see geographic setup guide)"
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Activate environment: source activate_fraud_env.sh"
    echo "   2. Test core ML: python test_core.py"
    echo "   3. Build your fraud detection system!"
    echo ""
    echo "🔧 To reactivate environment later:"
    if [[ "$setup_method" == "conda" ]]; then
        echo "   conda activate fraud_detection"
    else
        echo "   source fraud_detection_env/bin/activate"
    fi
    echo "   # OR"
    echo "   source activate_fraud_env.sh"
}

# Run main function
main