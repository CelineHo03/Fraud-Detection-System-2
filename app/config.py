
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
    
    # XGBoost parameters
    n_estimators: int = 100  # Reduced for faster testing
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
    """Simplified configuration manager"""
    
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

# Convenience functions for easy access (THESE WERE MISSING!)
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
