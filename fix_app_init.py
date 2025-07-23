#!/usr/bin/env python3
"""
Fix both app/__init__.py and app/config.py files
"""

import os
from pathlib import Path

def check_current_files():
    """Check what's in the current files."""
    print("🔍 Checking current files...")
    
    # Check app/__init__.py
    init_path = Path("app/__init__.py")
    if init_path.exists():
        with open(init_path, 'r') as f:
            init_content = f.read()
        print(f"app/__init__.py length: {len(init_content)} chars")
        if "from .config import" in init_content:
            print("❌ app/__init__.py has problematic config imports")
        else:
            print("✅ app/__init__.py looks ok")
    else:
        print("❌ app/__init__.py missing")
    
    # Check app/config.py
    config_path = Path("app/config.py")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_content = f.read()
        print(f"app/config.py length: {len(config_content)} chars")
        if "def get_config(" in config_content:
            print("✅ app/config.py has get_config function")
        else:
            print("❌ app/config.py missing get_config function")
    else:
        print("❌ app/config.py missing")

def fix_app_init():
    """Fix the app/__init__.py file to remove problematic imports."""
    print("\n🔧 Fixing app/__init__.py...")
    
    # Create a simple __init__.py that doesn't import config at module level
    init_content = '''"""
Fraud Detection System - Main Application Package

A production-ready fraud detection system that combines state-of-the-art 
machine learning with practical business applications.
"""

__version__ = "2.0.0"
__author__ = "Fraud Detection Team"
__description__ = "Production-ready fraud detection system with advanced ML capabilities"

# Don't import config at module level to avoid circular imports
# Import functions are available but must be imported explicitly when needed

__all__ = [
    "__version__",
    "__author__", 
    "__description__"
]
'''
    
    init_path = Path("app/__init__.py")
    with open(init_path, 'w') as f:
        f.write(init_content)
    
    print("✅ Fixed app/__init__.py (removed module-level config imports)")

def fix_config_file():
    """Fix the app/config.py file."""
    print("\n🔧 Fixing app/config.py...")
    
    config_content = '''"""
Configuration management for the Fraud Detection System.
Fixed version with all required functions.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Get the project root directory  
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TEMP_DIR = PROJECT_ROOT / "data" / "temp_uploads"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class ModelConfig:
    """Model configuration settings"""
    model_type: str = "xgboost"
    model_version: str = "2.0.0"
    use_advanced_model: bool = True
    model_path: str = str(MODELS_DIR / "saved_models")
    
    # XGBoost parameters (reduced for testing)
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Training parameters
    use_groupkfold: bool = True
    n_splits: int = 3  # Reduced for faster testing
    early_stopping_rounds: int = 50
    
    # Feature engineering
    use_advanced_features: bool = True
    feature_count: int = 263
    include_uid_feature: bool = True
    
    # Prediction thresholds
    default_threshold: float = 0.5
    high_precision_threshold: float = 0.7
    high_recall_threshold: float = 0.3

@dataclass
class StreamlitConfig:
    """Streamlit-specific configuration"""
    page_title: str = "Fraud Detection System"
    page_icon: str = "🔍"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # Upload settings
    max_upload_size_mb: int = 200
    allowed_file_types: list = None
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = ['csv', 'xlsx', 'xls']

@dataclass
class VisualizationConfig:
    """Visualization settings"""
    default_theme: str = "plotly_white"
    map_style: str = "OpenStreetMap"
    color_palette: list = None
    
    # Chart settings
    chart_height: int = 400
    map_height: int = 500
    
    # Performance settings
    max_points_on_map: int = 10000
    sampling_threshold: int = 50000
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = [
                '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD'
            ]

@dataclass
class DataConfig:
    """Data processing configuration"""
    # Paths
    ieee_data_path: str = str(DATA_DIR / "ieee")
    temp_upload_path: str = str(TEMP_DIR)
    cache_path: str = str(DATA_DIR / "cache")
    
    # Processing settings
    chunk_size: int = 1000  # Reduced for testing
    memory_limit_gb: float = 4.0
    
    # Validation settings
    min_required_columns: int = 3
    max_missing_percentage: float = 50.0
    
    # Dataset adaptation
    auto_detect_format: bool = True
    create_synthetic_features: bool = True

class ConfigManager:
    """Central configuration manager"""
    
    def __init__(self):
        self.app_name = os.getenv("APP_NAME", "fraud-detection-system")
        self.app_env = os.getenv("APP_ENV", "development")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"
        
        # Initialize configuration objects
        self.model = ModelConfig()
        self.streamlit = StreamlitConfig()
        self.visualization = VisualizationConfig()
        self.data = DataConfig()
    
    def get_model_file_paths(self) -> Dict[str, str]:
        """Get paths to model files"""
        base_path = Path(self.model.model_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        if self.model.use_advanced_model:
            return {
                'model': str(base_path / "xgb_model_advanced.pkl"),
                'feature_engineer': str(base_path / "feature_engineer_advanced.pkl"),
                'metadata': str(base_path / "metadata_advanced.json")
            }
        else:
            return {
                'model': str(base_path / "xgb_model.pkl"),
                'feature_engineer': str(base_path / "feature_engineer.pkl"),
                'metadata': str(base_path / "metadata.json")
            }
    
    def get_upload_settings(self) -> Dict[str, Any]:
        """Get upload-related settings for Streamlit"""
        return {
            'max_size_mb': self.streamlit.max_upload_size_mb,
            'allowed_types': self.streamlit.allowed_file_types,
            'temp_path': self.data.temp_upload_path
        }
    
    def get_visualization_settings(self) -> Dict[str, Any]:
        """Get visualization settings"""
        return {
            'theme': self.visualization.default_theme,
            'color_palette': self.visualization.color_palette,
            'chart_height': self.visualization.chart_height,
            'map_height': self.visualization.map_height,
            'map_style': self.visualization.map_style
        }
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.app_env == "development" or self.debug

# Global configuration instance
config = ConfigManager()

# Required functions that were missing!
def get_config() -> ConfigManager:
    """Get the global configuration instance"""
    return config

def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return config.model

def get_streamlit_config() -> StreamlitConfig:
    """Get Streamlit configuration"""
    return config.streamlit

def get_visualization_config() -> VisualizationConfig:
    """Get visualization configuration"""
    return config.visualization

def get_data_config() -> DataConfig:
    """Get data configuration"""
    return config.data

# Configuration validation
def validate_config() -> bool:
    """Validate that all required configurations are properly set"""
    try:
        # Check model paths exist
        model_paths = config.get_model_file_paths()
        models_dir = Path(config.model.model_path)
        if not models_dir.exists():
            models_dir.mkdir(parents=True, exist_ok=True)
        
        # Check data directories
        for path in [config.data.ieee_data_path, config.data.temp_upload_path, config.data.cache_path]:
            path_obj = Path(path)
            if not path_obj.exists():
                path_obj.mkdir(parents=True, exist_ok=True)
        
        return True
    except Exception as e:
        print(f"Warning: Configuration validation failed: {e}")
        return False

# Initialize and validate on import
validate_config()
'''
    
    config_path = Path("app/config.py")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print("✅ Fixed app/config.py (added all required functions)")

def test_imports():
    """Test if imports work now."""
    print("\n🧪 Testing imports...")
    
    try:
        # Test direct import
        from app.config import get_config, get_model_config, get_streamlit_config
        print("✅ Direct config imports successful!")
        
        # Test the functions work
        config = get_config()
        model_config = get_model_config()
        streamlit_config = get_streamlit_config()
        
        print(f"✅ Config manager: {type(config).__name__}")
        print(f"✅ Model config: {model_config.model_type}")
        print(f"✅ Streamlit config: {streamlit_config.page_title}")
        
        # Test app module import (this was failing before)
        import app
        print("✅ app module import successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Import still failing: {e}")
        return False

def main():
    """Main fix function."""
    print("🔧 Comprehensive Import Fix")
    print("=" * 40)
    
    # Check current state
    check_current_files()
    
    # Fix both files
    fix_app_init()
    fix_config_file()
    
    # Test imports
    success = test_imports()
    
    if success:
        print("\n🎉 SUCCESS! All imports are now working!")
        print("\n🎯 You can now run:")
        print("   python test_training_pipeline.py")
        print("\n📋 What was fixed:")
        print("   ✅ app/__init__.py - removed problematic module-level imports")
        print("   ✅ app/config.py - added all required functions")
        print("   ✅ All config functions now available")
    else:
        print("\n❌ Still having issues. Let's try one more approach...")
        print("Try running:")
        print("   python -c 'import sys; print(sys.path)'")
        print("   python -c 'import app.config; print(dir(app.config))'")

if __name__ == "__main__":
    main()