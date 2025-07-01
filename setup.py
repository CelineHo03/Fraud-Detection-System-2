#!/usr/bin/env python3
"""
Fraud Detection System - Project Setup Script
Creates the complete directory structure for the upload-centric fraud detection system.
"""

import os
from pathlib import Path

def create_project_structure():
    """Create the complete project directory structure"""
    
    # Define the project structure
    structure = {
        "": [
            "streamlit_app.py",
            "requirements.txt", 
            "README.md",
            ".gitignore",
            "Dockerfile",
            "docker-compose.yml"
        ],
        "pages": [
            "1_🔍_Upload_and_Analyze.py",
            "2_📊_Analytics_Dashboard.py", 
            "3_🌍_Geographic_Analysis.py",
            "4_📈_Performance_Metrics.py",
            "5_📋_Batch_Processing.py",
            "6_📁_Export_Results.py"
        ],
        "app": ["__init__.py"],
        "app/core": [
            "__init__.py",
            "session_manager.py",
            "upload_handler.py", 
            "cache_manager.py"
        ],
        "app/ml": [
            "__init__.py",
            "dataset_adapter.py",
            "predictor.py",
            "feature_engineer.py",
            "model_loader.py"
        ],
        "app/ui": [
            "__init__.py",
            "components.py",
            "charts.py",
            "maps.py", 
            "uploader.py",
            "styling.py"
        ],
        "app/analytics": [
            "__init__.py",
            "dashboard_generator.py",
            "insights.py",
            "reports.py",
            "exporters.py"
        ],
        "app/utils": [
            "__init__.py",
            "validators.py",
            "logger.py",
            "helpers.py",
            "performance.py"
        ],
        "models": [
            ".gitkeep"
        ],
        "data": [""],
        "data/ieee": [".gitkeep"],
        "data/temp_uploads": [".gitkeep"], 
        "data/cache": [".gitkeep"],
        "assets": [""],
        "assets/css": [".gitkeep"],
        "assets/images": [".gitkeep"],
        "scripts": [
            "setup_ieee_model.py",
            "validate_installation.py",
            "download_ieee_data.py"
        ],
        "tests": [
            "__init__.py",
            "conftest.py",
            "test_upload_processing.py",
            "test_ml_pipeline.py"
        ],
        "config": [
            "model_config.yaml",
            "logging.yaml"
        ],
        ".streamlit": [
            "config.toml",
            "secrets.toml.example"
        ]
    }
    
    # Create directories and files
    for directory, files in structure.items():
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {directory}")
        
        for file in files:
            if file:  # Skip empty strings
                file_path = Path(directory) / file if directory else Path(file)
                if not file_path.exists():
                    file_path.touch()
                    print(f"📄 Created file: {file_path}")

def create_requirements_txt():
    """Create requirements.txt with all necessary dependencies"""
    requirements = """# Core Streamlit and Web Framework
streamlit>=1.28.0
streamlit-folium>=0.15.0
streamlit-aggrid>=0.3.4

# Machine Learning and Data Science
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
joblib>=1.3.0

# Visualization
plotly>=5.17.0
seaborn>=0.12.0
matplotlib>=3.7.0
folium>=0.14.0

# Model Explainability
shap>=0.43.0

# Data Processing
openpyxl>=3.1.0
xlrd>=2.0.0
fastparquet>=0.8.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
loguru>=0.7.0
tqdm>=4.65.0

# Performance and Caching
redis>=4.5.0
psutil>=5.9.0

# Development and Testing
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("📋 Created requirements.txt")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# Data files
data/ieee/
data/temp_uploads/*
!data/temp_uploads/.gitkeep
data/cache/*
!data/cache/.gitkeep

# Model files
models/*.pkl
models/*.joblib
models/*.json
!models/.gitkeep

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml

# Docker
docker-data/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("🚫 Created .gitignore")

def create_streamlit_config():
    """Create Streamlit configuration files"""
    
    # Main config
    config_content = """[global]
developmentMode = false

[server]
runOnSave = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"  
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
"""
    
    Path(".streamlit").mkdir(exist_ok=True)
    with open(".streamlit/config.toml", "w") as f:
        f.write(config_content)
    
    # Secrets template
    secrets_template = """# Streamlit Secrets Configuration
# Copy this file to secrets.toml and fill in your values

[database]
# Database connection (optional)
# DATABASE_URL = "postgresql://user:pass@localhost/fraud_db"

[api]
# API keys for external services (optional)
# GOOGLE_MAPS_API_KEY = "your_google_maps_key"
# OPENAI_API_KEY = "your_openai_key"

[model]
# Model configuration
USE_ADVANCED_MODEL = true
MODEL_VERSION = "2.0.0"

[cache]
# Cache settings
CACHE_TTL = 3600
USE_REDIS = false
# REDIS_URL = "redis://localhost:6379/0"
"""
    
    with open(".streamlit/secrets.toml.example", "w") as f:
        f.write(secrets_template)
    
    print("⚙️ Created Streamlit configuration files")

if __name__ == "__main__":
    print("🚀 Setting up Fraud Detection System project structure...")
    print()
    
    create_project_structure()
    print()
    
    create_requirements_txt()
    create_gitignore() 
    create_streamlit_config()
    
    print()
    print("✅ Project structure created successfully!")
    print()
    print("Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Copy .streamlit/secrets.toml.example to .streamlit/secrets.toml")
    print("3. Run the setup: python scripts/setup_ieee_model.py")
    print("4. Start the app: streamlit run streamlit_app.py")