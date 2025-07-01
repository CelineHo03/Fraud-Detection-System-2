"""
Fraud Detection System - Main Application Package

A production-ready fraud detection system that combines state-of-the-art 
machine learning with practical business applications.

Core Features:
- Advanced ML Model: IEEE competition-winning approach with 263 engineered features
- Real-time Predictions: Sub-50ms predictions with explainability
- Cross-Dataset Support: Train on IEEE, predict on any fraud dataset
- Geographic Visualization: Interactive maps showing fraud patterns
- Comprehensive Dashboard: Streamlit UI for analysis and monitoring
"""

__version__ = "2.0.0"
__author__ = "Fraud Detection Team"
__description__ = "Production-ready fraud detection system with advanced ML capabilities"

# Package-level imports for easy access
from .config import get_config, get_model_config, get_streamlit_config

__all__ = [
    "__version__",
    "__author__", 
    "__description__",
    "get_config",
    "get_model_config", 
    "get_streamlit_config"
]