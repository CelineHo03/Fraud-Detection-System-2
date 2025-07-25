"""
Advanced Prediction Engine for IEEE Fraud Detection
Sub-50ms predictions with SHAP explanations and risk scoring.

This module provides:
- Fast predictions using trained XGBoost model
- SHAP explanations for interpretability
- Risk scoring (0-100 scale)
- Batch prediction support
- Cross-dataset compatibility
- Threshold-based classifications
- Performance monitoring
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import shap
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# Local imports
from ..config import get_model_config, get_data_config
from .preprocessing_advanced import AdvancedFeatureEngineer


@dataclass
class PredictionResult:
    """Container for prediction results."""
    transaction_id: Optional[str] = None
    fraud_probability: float = 0.0
    risk_score: int = 0  # 0-100 scale
    is_fraud: bool = False
    threshold_used: float = 0.5
    confidence: str = "medium"  # low, medium, high
    shap_values: Optional[Dict[str, float]] = None
    top_risk_factors: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: float = 0.0


class AdvancedPredictor:
    """
    Advanced prediction engine for fraud detection.
    
    Features:
    - Sub-50ms prediction latency
    - SHAP explanations for every prediction
    - Risk scoring on 0-100 scale
    - Batch prediction support
    - Multiple threshold options
    - Performance monitoring
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 enable_shap: bool = True,
                 shap_sample_size: int = 100,
                 verbose: bool = False):
        """
        Initialize the advanced predictor.
        
        Args:
            model_path: Path to saved model artifacts
            enable_shap: Whether to compute SHAP explanations
            shap_sample_size: Background sample size for SHAP
            verbose: Whether to print loading information
        """
        self.enable_shap = enable_shap
        self.shap_sample_size = shap_sample_size
        self.verbose = verbose
        
        # Get configuration
        self.model_config = get_model_config()
        self.data_config = get_data_config()
        
        # Set model path
        if model_path is None:
            model_path = Path(self.model_config.model_path)
        self.model_path = Path(model_path)
        
        # Initialize components
        self.model = None
        self.feature_engineer = None
        self.metadata = {}
        self.shap_explainer = None
        self.feature_names = []
        self.is_loaded = False
        
        # Performance tracking
        self.prediction_times = []
        
        # Load model artifacts
        self._load_model_artifacts()
    
    def _load_model_artifacts(self):
        """Load all model artifacts."""
        if self.verbose:
            print("Loading model artifacts...")
        
        start_time = time.time()
        
        # Load trained model
        model_file = self.model_path / "xgb_model_advanced.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        self.model = joblib.load(model_file)
        
        # Load feature engineer
        engineer_file = self.model_path / "feature_engineer_advanced.pkl"
        if not engineer_file.exists():
            raise FileNotFoundError(f"Feature engineer not found: {engineer_file}")
        
        self.feature_engineer = joblib.load(engineer_file)
        self.feature_names = self.feature_engineer.get_feature_names()
        
        # Load metadata
        metadata_file = self.model_path / "metadata_advanced.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        
        # Initialize SHAP explainer
        if self.enable_shap:
            self._initialize_shap_explainer()
        
        loading_time = time.time() - start_time
        self.is_loaded = True
        
        if self.verbose:
            print(f"Model loaded successfully ({loading_time:.2f}s)")
            print(f"Model version: {self.metadata.get('model_version', 'unknown')}")
            print(f"Feature count: {len(self.feature_names)}")
            if self.enable_shap:
                print("SHAP explainer initialized")
    
    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer for model interpretability."""
        if self.verbose:
            print("Initializing SHAP explainer...")
        
        try:
            # Create a background dataset for SHAP
            # Note: In production, you'd want to use a representative sample
            # from your training data. For now, we'll create a placeholder.
            background_data = np.zeros((self.shap_sample_size, len(self.feature_names)))
            
            # Initialize TreeExplainer for XGBoost
            self.shap_explainer = shap.TreeExplainer(self.model, background_data)
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not initialize SHAP explainer: {e}")
            self.enable_shap = False
            self.shap_explainer = None
    
    def predict_single(self, 
                      transaction_data: Union[pd.DataFrame, Dict],
                      transaction_id: Optional[str] = None,
                      threshold: Optional[float] = None,
                      include_shap: Optional[bool] = None) -> PredictionResult:
        """
        Predict fraud probability for a single transaction.
        
        Args:
            transaction_data: Transaction data (DataFrame or dict)
            transaction_id: Optional transaction identifier
            threshold: Classification threshold (defaults to model default)
            include_shap: Whether to compute SHAP values
            
        Returns:
            PredictionResult with fraud probability, risk score, and explanations
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before making predictions")
        
        start_time = time.time()
        
        # Convert to DataFrame if needed
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data.copy()
        
        # Set defaults
        if threshold is None:
            threshold = self.model_config.default_threshold
        if include_shap is None:
            include_shap = self.enable_shap
        
        # Feature engineering
        try:
            X_features = self.feature_engineer.transform(df)
        except Exception as e:
            raise ValueError(f"Feature engineering failed: {e}")
        
        # Make prediction
        fraud_prob = self.model.predict_proba(X_features)[0, 1]
        is_fraud = fraud_prob >= threshold
        
        # Calculate risk score (0-100)
        risk_score = int(fraud_prob * 100)
        
        # Determine confidence level
        confidence = self._calculate_confidence(fraud_prob, threshold)
        
        # SHAP explanations
        shap_values = None
        top_risk_factors = None
        
        if include_shap and self.shap_explainer is not None:
            try:
                shap_vals = self.shap_explainer.shap_values(X_features)[0]
                
                # Create SHAP value dictionary
                shap_values = dict(zip(self.feature_names, shap_vals))
                
                # Get top risk factors
                top_risk_factors = self._get_top_risk_factors(shap_values, X_features.iloc[0])
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: SHAP calculation failed: {e}")
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        self.prediction_times.append(processing_time)
        
        # Create result
        result = PredictionResult(
            transaction_id=transaction_id,
            fraud_probability=float(fraud_prob),
            risk_score=risk_score,
            is_fraud=is_fraud,
            threshold_used=threshold,
            confidence=confidence,
            shap_values=shap_values,
            top_risk_factors=top_risk_factors,
            processing_time_ms=processing_time
        )
        
        return result
    
    def predict_batch(self, 
                     transactions_df: pd.DataFrame,
                     id_column: Optional[str] = 'TransactionID',
                     threshold: Optional[float] = None,
                     include_shap: bool = False,
                     chunk_size: Optional[int] = None) -> List[PredictionResult]:
        """
        Predict fraud probabilities for multiple transactions.
        
        Args:
            transactions_df: DataFrame with transaction data
            id_column: Column name for transaction IDs
            threshold: Classification threshold
            include_shap: Whether to compute SHAP values
            chunk_size: Process in chunks (for memory efficiency)
            
        Returns:
            List of PredictionResult objects
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before making predictions")
        
        if self.verbose:
            print(f"Batch prediction for {len(transactions_df)} transactions")
        
        start_time = time.time()
        
        # Set defaults
        if threshold is None:
            threshold = self.model_config.default_threshold
        if chunk_size is None:
            chunk_size = self.data_config.chunk_size
        
        results = []
        
        # Process in chunks
        for i in range(0, len(transactions_df), chunk_size):
            chunk = transactions_df.iloc[i:i+chunk_size]
            
            # Feature engineering
            try:
                X_features = self.feature_engineer.transform(chunk)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Feature engineering failed for chunk {i}: {e}")
                continue
            
            # Make predictions
            fraud_probs = self.model.predict_proba(X_features)[:, 1]
            
            # SHAP explanations (if requested)
            shap_values_batch = None
            if include_shap and self.shap_explainer is not None:
                try:
                    shap_values_batch = self.shap_explainer.shap_values(X_features)
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: SHAP calculation failed for chunk {i}: {e}")
            
            # Create results for chunk
            for j, (idx, row) in enumerate(chunk.iterrows()):
                # Get transaction ID
                trans_id = None
                if id_column and id_column in chunk.columns:
                    trans_id = str(row[id_column])
                
                fraud_prob = fraud_probs[j]
                
                # SHAP values for this transaction
                shap_values = None
                top_risk_factors = None
                if shap_values_batch is not None:
                    shap_vals = shap_values_batch[j]
                    shap_values = dict(zip(self.feature_names, shap_vals))
                    top_risk_factors = self._get_top_risk_factors(shap_values, X_features.iloc[j])
                
                result = PredictionResult(
                    transaction_id=trans_id,
                    fraud_probability=float(fraud_prob),
                    risk_score=int(fraud_prob * 100),
                    is_fraud=fraud_prob >= threshold,
                    threshold_used=threshold,
                    confidence=self._calculate_confidence(fraud_prob, threshold),
                    shap_values=shap_values,
                    top_risk_factors=top_risk_factors,
                    processing_time_ms=0.0  # Will be calculated after batch
                )
                
                results.append(result)
        
        # Calculate average processing time
        total_time = (time.time() - start_time) * 1000
        avg_time_per_transaction = total_time / len(results) if results else 0
        
        # Update processing times
        for result in results:
            result.processing_time_ms = avg_time_per_transaction
        
        if self.verbose:
            print(f"Batch prediction complete: {len(results)} results in {total_time:.2f}ms")
            print(f"Average time per transaction: {avg_time_per_transaction:.2f}ms")
        
        return results
    
    def _calculate_confidence(self, fraud_prob: float, threshold: float) -> str:
        """Calculate confidence level based on probability and threshold."""
        distance_from_threshold = abs(fraud_prob - threshold)
        
        if distance_from_threshold >= 0.3:
            return "high"
        elif distance_from_threshold >= 0.1:
            return "medium"
        else:
            return "low"
    
    
    
    def _get_top_risk_factors(self, shap_values: Dict[str, float], feature_values: pd.Series, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get top risk factors from SHAP values."""
        # Sort SHAP values by absolute magnitude
        sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        
        risk_factors = []
        for feature_name, shap_value in sorted_shap[:top_n]:
            feature_value = feature_values[feature_name] if feature_name in feature_values.index else None
            
            risk_factors.append({
                'feature': feature_name,
                'shap_value': float(shap_value),
                'feature_value': float(feature_value) if feature_value is not None else None,
                'impact': 'increases_risk' if shap_value > 0 else 'decreases_risk',
                'importance': abs(float(shap_value))
            })
        
        return risk_factors
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get model feature importance."""
        if not self.is_loaded:
            raise ValueError("Model must be loaded first")
        
        # Get feature importance from model
        importance_scores = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Sort by importance
        sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_importance[:top_n])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata."""
        info = {
            'is_loaded': self.is_loaded,
            'model_path': str(self.model_path),
            'feature_count': len(self.feature_names),
            'shap_enabled': self.enable_shap,
            'metadata': self.metadata
        }
        
        if self.prediction_times:
            info['performance'] = {
                'predictions_made': len(self.prediction_times),
                'avg_prediction_time_ms': np.mean(self.prediction_times),
                'min_prediction_time_ms': np.min(self.prediction_times),
                'max_prediction_time_ms': np.max(self.prediction_times)
            }
        
        return info
    
    def predict_with_multiple_thresholds(self, 
                                       transaction_data: Union[pd.DataFrame, Dict],
                                       thresholds: Optional[List[float]] = None) -> Dict[str, PredictionResult]:
        """
        Predict with multiple classification thresholds.
        
        Args:
            transaction_data: Transaction data
            thresholds: List of thresholds to use
            
        Returns:
            Dictionary mapping threshold names to PredictionResult
        """
        if thresholds is None:
            thresholds = [
                self.model_config.high_recall_threshold,
                self.model_config.default_threshold,
                self.model_config.high_precision_threshold
            ]
        
        # Get base prediction (with SHAP)
        base_result = self.predict_single(transaction_data, include_shap=True)
        
        # Create results for different thresholds
        results = {}
        threshold_names = ['high_recall', 'default', 'high_precision']
        
        for i, threshold in enumerate(thresholds):
            name = threshold_names[i] if i < len(threshold_names) else f'threshold_{threshold}'
            
            # Create result with different threshold
            result = PredictionResult(
                transaction_id=base_result.transaction_id,
                fraud_probability=base_result.fraud_probability,
                risk_score=base_result.risk_score,
                is_fraud=base_result.fraud_probability >= threshold,
                threshold_used=threshold,
                confidence=self._calculate_confidence(base_result.fraud_probability, threshold),
                shap_values=base_result.shap_values,
                top_risk_factors=base_result.top_risk_factors,
                processing_time_ms=base_result.processing_time_ms
            )
            
            results[name] = result
        
        return results


def create_predictor(model_path: Optional[str] = None, **kwargs) -> AdvancedPredictor:
    """
    Convenience function to create and load a predictor.
    
    Args:
        model_path: Path to model artifacts
        **kwargs: Additional arguments for AdvancedPredictor
        
    Returns:
        Loaded AdvancedPredictor instance
    """
    return AdvancedPredictor(model_path=model_path, **kwargs)


def predict_fraud(transaction_data: Union[pd.DataFrame, Dict], 
                  model_path: Optional[str] = None,
                  **kwargs) -> PredictionResult:
    """
    Quick fraud prediction for a single transaction.
    
    Args:
        transaction_data: Transaction data
        model_path: Path to model artifacts
        **kwargs: Additional arguments
        
    Returns:
        PredictionResult
    """
    predictor = create_predictor(model_path=model_path, verbose=False)
    return predictor.predict_single(transaction_data, **kwargs)


# Example usage
if __name__ == "__main__":
    print("Advanced Prediction Engine for IEEE Fraud Detection")
    print("Usage:")
    print("  from app.ml.predictor import create_predictor")
    print("  predictor = create_predictor()")
    print("  result = predictor.predict_single(transaction_data)")
    print("\nFeatures:")
    print("  - Sub-50ms predictions")
    print("  - SHAP explanations")
    print("  - Risk scoring (0-100)")
    print("  - Batch predictions")
    print("  - Multiple thresholds")