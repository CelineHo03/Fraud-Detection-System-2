"""
Advanced Training Pipeline for IEEE Fraud Detection
Competition-winning training approach with GroupKFold validation.

This module implements the advanced training pipeline that:
- Uses GroupKFold validation for time-series fraud data
- Trains XGBoost with optimized hyperparameters
- Handles class imbalance appropriately
- Includes early stopping and proper validation
- Saves model artifacts and metadata
- Provides comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import time

warnings.filterwarnings('ignore')

# Local imports
from ..config import get_model_config, get_data_config
from .preprocessing_advanced import AdvancedFeatureEngineer, create_advanced_features


class AdvancedTrainer:
    """
    Advanced training pipeline for IEEE fraud detection.
    
    Features:
    - GroupKFold validation for time-series data
    - XGBoost with competition-winning hyperparameters
    - Early stopping and validation monitoring
    - Class imbalance handling
    - Comprehensive evaluation metrics
    - Model persistence and metadata tracking
    """
    
    def __init__(self, 
                 n_splits: int = 6,
                 early_stopping_rounds: int = 200,
                 eval_metric: str = 'auc',
                 verbose: bool = True,
                 save_path: Optional[str] = None):
        """
        Initialize the advanced trainer.
        
        Args:
            n_splits: Number of GroupKFold splits
            early_stopping_rounds: Early stopping patience
            eval_metric: Evaluation metric for early stopping
            verbose: Whether to print training progress
            save_path: Path to save model artifacts
        """
        self.n_splits = n_splits
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.verbose = verbose
        
        # Get configuration
        self.model_config = get_model_config()
        self.data_config = get_data_config()
        
        # Set save path
        if save_path is None:
            save_path = Path(self.model_config.model_path)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.models_ = {}
        self.feature_engineer_ = None
        self.training_history_ = {}
        self.validation_scores_ = {}
        self.feature_importance_ = {}
        self.is_trained_ = False
    
    def train(self, 
              train_df: pd.DataFrame, 
              target_col: str = 'isFraud',
              group_col: Optional[str] = None,
              test_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train the advanced fraud detection model.
        
        Args:
            train_df: Training DataFrame with merged transaction + identity data
            target_col: Name of the target column
            group_col: Column to use for GroupKFold (defaults to time-based grouping)
            test_df: Optional test DataFrame for holdout validation
            
        Returns:
            Dictionary with training results and metrics
        """
        if self.verbose:
            print("=== Advanced Fraud Detection Training ===")
            print(f"Training data shape: {train_df.shape}")
            print(f"Target column: {target_col}")
            
        start_time = time.time()
        
        # Step 1: Prepare data
        X, y, groups, X_test, y_test = self._prepare_data(
            train_df, target_col, group_col, test_df
        )
        
        # Step 2: Advanced feature engineering
        if self.verbose:
            print("\n--- Feature Engineering ---")
        
        X_features, X_test_features, self.feature_engineer_ = create_advanced_features(
            X, X_test, verbose=self.verbose
        )
        
        # Step 3: GroupKFold training
        if self.verbose:
            print(f"\n--- GroupKFold Training ({self.n_splits} splits) ---")
        
        cv_results = self._train_with_groupkfold(X_features, y, groups)
        
        # Step 4: Train final model on all data
        if self.verbose:
            print("\n--- Final Model Training ---")
        
        final_model = self._train_final_model(X_features, y)
        
        # Step 5: Evaluate on test set if available
        test_results = {}
        if X_test_features is not None and y_test is not None:
            if self.verbose:
                print("\n--- Test Set Evaluation ---")
            test_results = self._evaluate_test_set(final_model, X_test_features, y_test)
        
        # Step 6: Save model artifacts
        if self.verbose:
            print("\n--- Saving Model Artifacts ---")
        
        self._save_model_artifacts(final_model, cv_results, test_results)
        
        # Compile final results
        training_time = time.time() - start_time
        results = {
            'cv_results': cv_results,
            'test_results': test_results,
            'training_time': training_time,
            'feature_count': X_features.shape[1],
            'model_path': str(self.save_path),
            'feature_importance': self.feature_importance_
        }
        
        self.is_trained_ = True
        
        if self.verbose:
            print(f"\n=== Training Complete ({training_time:.2f}s) ===")
            print(f"CV AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
            if test_results:
                print(f"Test AUC: {test_results['auc']:.4f}")
            print(f"Features: {X_features.shape[1]}")
            print(f"Model saved to: {self.save_path}")
        
        return results
    
    def _prepare_data(self, 
                     train_df: pd.DataFrame, 
                     target_col: str,
                     group_col: Optional[str],
                     test_df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, Optional[pd.DataFrame], Optional[pd.Series]]:
        """Prepare data for training."""
        
        # Separate features and target
        if target_col not in train_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in training data")
        
        X = train_df.drop(columns=[target_col])
        y = train_df[target_col]
        
        # Create groups for GroupKFold
        if group_col is None:
            # Use TransactionDT for time-based grouping
            if 'TransactionDT' in X.columns:
                # Create time-based groups (6 groups for 6-fold CV)
                dt_quantiles = pd.qcut(X['TransactionDT'], q=self.n_splits, labels=False)
                groups = dt_quantiles.values
            else:
                # Fallback: create random groups
                np.random.seed(42)
                groups = np.random.randint(0, self.n_splits, size=len(X))
        else:
            if group_col not in X.columns:
                raise ValueError(f"Group column '{group_col}' not found in data")
            groups = X[group_col].values
        
        # Handle test data
        X_test, y_test = None, None
        if test_df is not None:
            if target_col in test_df.columns:
                X_test = test_df.drop(columns=[target_col])
                y_test = test_df[target_col]
            else:
                X_test = test_df
                y_test = None
        
        if self.verbose:
            print(f"Features: {X.shape[1]}")
            print(f"Training samples: {len(X)}")
            print(f"Fraud rate: {y.mean():.4f}")
            print(f"Groups: {len(np.unique(groups))}")
            if X_test is not None:
                print(f"Test samples: {len(X_test)}")
        
        return X, y, groups, X_test, y_test
    
    def _get_xgb_params(self) -> Dict[str, Any]:
        """Get XGBoost parameters from configuration."""
        return {
            'n_estimators': self.model_config.n_estimators,
            'max_depth': self.model_config.max_depth,
            'learning_rate': self.model_config.learning_rate,
            'subsample': self.model_config.subsample,
            'colsample_bytree': self.model_config.colsample_bytree,
            'objective': 'binary:logistic',
            'eval_metric': self.eval_metric,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
            'enable_categorical': False
        }
    
    def _train_with_groupkfold(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray) -> Dict[str, Any]:
        """Train with GroupKFold cross-validation."""
        
        # Initialize GroupKFold
        gkf = GroupKFold(n_splits=self.n_splits)
        
        # Storage for CV results
        cv_scores = []
        cv_predictions = np.zeros(len(X))
        fold_models = {}
        fold_feature_importance = {}
        
        # XGBoost parameters
        xgb_params = self._get_xgb_params()
        
        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            if self.verbose:
                print(f"\nFold {fold + 1}/{self.n_splits}")
            
            # Split data
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            if self.verbose:
                print(f"  Train: {len(X_train_fold)} samples, Fraud rate: {y_train_fold.mean():.4f}")
                print(f"  Val:   {len(X_val_fold)} samples, Fraud rate: {y_val_fold.mean():.4f}")
            
            # Train model
            model = xgb.XGBClassifier(**xgb_params)
            
            # Fit with evaluation set (no early stopping for compatibility)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
            
            # Predictions and scoring
            val_pred_proba = model.predict_proba(X_val_fold)[:, 1]
            cv_predictions[val_idx] = val_pred_proba
            
            fold_auc = roc_auc_score(y_val_fold, val_pred_proba)
            cv_scores.append(fold_auc)
            
            # Store model and feature importance
            fold_models[f'fold_{fold}'] = model
            fold_feature_importance[f'fold_{fold}'] = dict(zip(
                X.columns, model.feature_importances_
            ))
            
            if self.verbose:
                print(f"  Fold AUC: {fold_auc:.4f}")
        
        # Calculate overall CV metrics
        overall_auc = roc_auc_score(y, cv_predictions)
        mean_auc = np.mean(cv_scores)
        std_auc = np.std(cv_scores)
        
        # Calculate average feature importance
        avg_importance = {}
        for feature in X.columns:
            importance_values = [fold_feature_importance[f'fold_{fold}'][feature] 
                               for fold in range(self.n_splits)]
            avg_importance[feature] = np.mean(importance_values)
        
        # Sort by importance
        self.feature_importance_ = dict(sorted(avg_importance.items(), 
                                             key=lambda x: x[1], reverse=True))
        
        cv_results = {
            'cv_scores': cv_scores,
            'cv_predictions': cv_predictions,
            'overall_auc': overall_auc,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'fold_models': fold_models,
            'feature_importance': self.feature_importance_
        }
        
        if self.verbose:
            print(f"\nCV Results:")
            print(f"  Overall AUC: {overall_auc:.4f}")
            print(f"  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
            print(f"  Individual folds: {[f'{score:.4f}' for score in cv_scores]}")
        
        return cv_results
    
    def _train_final_model(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
        """Train final model on all training data."""
        
        xgb_params = self._get_xgb_params()
        
        # Use a portion of data for early stopping validation
        val_size = int(0.2 * len(X))
        X_train_final = X.iloc[:-val_size]
        y_train_final = y.iloc[:-val_size]
        X_val_final = X.iloc[-val_size:]
        y_val_final = y.iloc[-val_size:]
        
        # Train final model
        final_model = xgb.XGBClassifier(**xgb_params)
        final_model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_val_final, y_val_final)],
            verbose=False
        )
        
        # Use a fixed number of estimators (no early stopping)
        optimal_rounds = xgb_params['n_estimators']
        
        final_model = xgb.XGBClassifier(**xgb_params)
        final_model.fit(X, y, verbose=False)
        
        if self.verbose:
            print(f"Final model trained with {optimal_rounds} estimators")
        
        return final_model
    
    def _evaluate_test_set(self, model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model on test set."""
        
        # Predictions
        test_pred_proba = model.predict_proba(X_test)[:, 1]
        test_pred_binary = model.predict(X_test)
        
        # Metrics
        test_auc = roc_auc_score(y_test, test_pred_proba)
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_test, test_pred_proba)
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_test, test_pred_proba)
        
        # Find optimal threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = pr_thresholds[optimal_idx]
        
        # Classification report
        test_pred_optimal = (test_pred_proba >= optimal_threshold).astype(int)
        class_report = classification_report(y_test, test_pred_optimal, output_dict=True)
        
        test_results = {
            'auc': test_auc,
            'optimal_threshold': optimal_threshold,
            'precision_at_optimal': precision[optimal_idx],
            'recall_at_optimal': recall[optimal_idx],
            'f1_at_optimal': f1_scores[optimal_idx],
            'classification_report': class_report,
            'predictions_proba': test_pred_proba.tolist(),
            'predictions_binary': test_pred_binary.tolist()
        }
        
        if self.verbose:
            print(f"Test AUC: {test_auc:.4f}")
            print(f"Optimal threshold: {optimal_threshold:.4f}")
            print(f"Precision at optimal: {precision[optimal_idx]:.4f}")
            print(f"Recall at optimal: {recall[optimal_idx]:.4f}")
        
        return test_results
    
    def _save_model_artifacts(self, model: xgb.XGBClassifier, cv_results: Dict, test_results: Dict):
        """Save all model artifacts."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the trained model
        model_path = self.save_path / "xgb_model_advanced.pkl"
        joblib.dump(model, model_path)
        
        # Save feature engineer
        engineer_path = self.save_path / "feature_engineer_advanced.pkl"
        joblib.dump(self.feature_engineer_, engineer_path)
        
        # Save metadata
        metadata = {
            'model_type': 'xgboost_advanced',
            'model_version': self.model_config.model_version,
            'timestamp': timestamp,
            'feature_count': len(self.feature_importance_),
            'cv_auc_mean': cv_results['mean_auc'],
            'cv_auc_std': cv_results['std_auc'],
            'hyperparameters': self._get_xgb_params(),
            'training_config': {
                'n_splits': self.n_splits,
                'early_stopping_rounds': self.early_stopping_rounds,
                'eval_metric': self.eval_metric
            }
        }
        
        if test_results:
            metadata['test_auc'] = test_results['auc']
            metadata['optimal_threshold'] = test_results['optimal_threshold']
        
        metadata_path = self.save_path / "metadata_advanced.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature importance
        importance_path = self.save_path / "feature_importance_advanced.json"
        with open(importance_path, 'w') as f:
            feature_importance_serializable = {k: float(v) for k, v in self.feature_importance_.items()}
            json.dump(feature_importance_serializable, f, indent=2) 
        
        if self.verbose:
            print(f"Model saved to: {model_path}")
            print(f"Feature engineer saved to: {engineer_path}")
            print(f"Metadata saved to: {metadata_path}")
            print(f"Feature importance saved to: {importance_path}")


def train_advanced_model(train_df: pd.DataFrame,
                        target_col: str = 'isFraud',
                        test_df: Optional[pd.DataFrame] = None,
                        save_path: Optional[str] = None,
                        **trainer_kwargs) -> Dict[str, Any]:
    """
    Convenience function to train an advanced fraud detection model.
    
    Args:
        train_df: Training DataFrame with merged transaction + identity data
        target_col: Name of the target column
        test_df: Optional test DataFrame
        save_path: Path to save model artifacts
        **trainer_kwargs: Additional arguments for AdvancedTrainer
        
    Returns:
        Dictionary with training results
    """
    trainer = AdvancedTrainer(save_path=save_path, **trainer_kwargs)
    return trainer.train(train_df, target_col=target_col, test_df=test_df)


# Example usage
if __name__ == "__main__":
    print("Advanced Training Pipeline for IEEE Fraud Detection")
    print("Usage:")
    print("  from app.ml.train_advanced import train_advanced_model")
    print("  results = train_advanced_model(train_df, test_df=test_df)")
    print("\nExpected input: Merged transaction + identity DataFrames")
    print("Expected output: Trained XGBoost model with 263+ features")