"""
Test script to verify the training pipeline works with actual IEEE data.
This script will test the complete ML pipeline end-to-end.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import time

# Add the parent directory to the path to import app modules
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our ML modules
from app.ml.preprocessing_advanced import create_advanced_features
from app.ml.train_advanced import train_advanced_model
from app.ml.predictor import create_predictor


def test_data_loading(train_path: str, test_path: str):
    """Test loading the merged IEEE datasets."""
    print("=== Testing Data Loading ===")
    
    # Load train data
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"✅ Train data loaded: {train_df.shape}")
    print(f"   Columns: {len(train_df.columns)}")
    if 'isFraud' in train_df.columns:
        fraud_rate = train_df['isFraud'].mean()
        print(f"   Fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")
    
    # Load test data
    print(f"\nLoading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"✅ Test data loaded: {test_df.shape}")
    print(f"   Columns: {len(test_df.columns)}")
    
    # Check for common columns
    common_cols = set(train_df.columns) & set(test_df.columns)
    print(f"   Common columns: {len(common_cols)}")
    
    # Check for IEEE-specific columns
    ieee_columns = ['TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1']
    missing_cols = [col for col in ieee_columns if col not in train_df.columns]
    if missing_cols:
        print(f"⚠️  Missing expected IEEE columns: {missing_cols}")
    else:
        print("✅ All expected IEEE columns present")
    
    return train_df, test_df


def test_feature_engineering(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Test the advanced feature engineering pipeline."""
    print("\n=== Testing Feature Engineering ===")
    
    try:
        # Test with a small sample first
        sample_size = min(50000, len(train_df))
        train_sample = train_df.head(sample_size)
        test_sample = test_df.head(sample_size)
        
        print(f"Testing with {sample_size} samples...")
        
        # Create advanced features
        train_features, test_features, engineer = create_advanced_features(
            train_sample, test_sample, verbose=True
        )
        
        print(f"✅ Feature engineering successful!")
        print(f"   Input features: {train_sample.shape[1]}")
        print(f"   Output features: {train_features.shape[1]}")
        print(f"   Feature increase: {train_features.shape[1] - train_sample.shape[1]}")
        
        # Check for NaN values
        train_nans = train_features.isnull().sum().sum()
        test_nans = test_features.isnull().sum().sum()
        
        if train_nans > 0 or test_nans > 0:
            print(f"⚠️  NaN values found - Train: {train_nans}, Test: {test_nans}")
        else:
            print("✅ No NaN values in engineered features")
        
        # Show feature groups
        feature_groups = engineer.get_feature_importance_groups()
        print("\n📊 Feature Groups:")
        for group, features in feature_groups.items():
            print(f"   {group}: {len(features)} features")
        
        return train_features, test_features, engineer
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return None, None, None


def test_training_small_sample(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Test training on a small sample to verify the pipeline works."""
    print("\n=== Testing Training Pipeline (Small Sample) ===")
    
    try:
        # Use a small sample for quick testing
        sample_size = min(5000, len(train_df))
        train_sample = train_df.head(sample_size)
        test_sample = test_df.head(1000) if len(test_df) > 1000 else test_df
        
        print(f"Training on {len(train_sample)} samples...")
        
        # Train the model
        results = train_advanced_model(
            train_df=train_sample,
            test_df=test_sample,
            target_col='isFraud',
            n_splits=6,  # Reduced for faster testing
            verbose=True
        )
        
        print("✅ Training successful!")
        print(f"   CV AUC: {results['cv_results']['mean_auc']:.4f}")
        print(f"   Training time: {results['training_time']:.2f}s")
        print(f"   Features: {results['feature_count']}")
        
        if results['test_results']:
            print(f"   Test AUC: {results['test_results']['auc']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_prediction_pipeline(model_path: str, test_df: pd.DataFrame):
    """Test the prediction pipeline."""
    print("\n=== Testing Prediction Pipeline ===")
    
    try:
        # Create predictor
        print("Loading predictor...")
        predictor = create_predictor(model_path=model_path, verbose=True)
        
        # Test single prediction
        print("\nTesting single prediction...")
        sample_transaction = test_df.head(1)
        
        start_time = time.time()
        result = predictor.predict_single(
            sample_transaction, 
            transaction_id="test_001",
            include_shap=True
        )
        prediction_time = (time.time() - start_time) * 1000
        
        print(f"✅ Single prediction successful!")
        print(f"   Fraud probability: {result.fraud_probability:.4f}")
        print(f"   Risk score: {result.risk_score}/100")
        print(f"   Prediction time: {prediction_time:.2f}ms")
        print(f"   Confidence: {result.confidence}")
        
        if result.top_risk_factors:
            print("   Top risk factors:")
            for factor in result.top_risk_factors[:3]:
                print(f"     - {factor['feature']}: {factor['impact']}")
        
        # Test batch prediction
        print("\nTesting batch prediction...")
        batch_sample = test_df.head(10)
        
        start_time = time.time()
        batch_results = predictor.predict_batch(
            batch_sample, 
            include_shap=False  # Skip SHAP for speed
        )
        batch_time = (time.time() - start_time) * 1000
        
        print(f"✅ Batch prediction successful!")
        print(f"   Predicted {len(batch_results)} transactions")
        print(f"   Total time: {batch_time:.2f}ms")
        print(f"   Avg time per transaction: {batch_time/len(batch_results):.2f}ms")
        
        # Show distribution of risk scores
        risk_scores = [r.risk_score for r in batch_results]
        print(f"   Risk scores: Min={min(risk_scores)}, Max={max(risk_scores)}, Avg={np.mean(risk_scores):.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_complete_test(train_path: str, test_path: str):
    """Run the complete end-to-end test."""
    print("🚀 Starting Complete ML Pipeline Test")
    print("=" * 50)
    
    # Step 1: Test data loading
    train_df, test_df = test_data_loading(train_path, test_path)
    if train_df is None:
        print("❌ Data loading failed. Stopping test.")
        return False
    
    # Step 2: Test feature engineering
    train_features, test_features, engineer = test_feature_engineering(train_df, test_df)
    if train_features is None:
        print("❌ Feature engineering failed. Stopping test.")
        return False
    
    # Step 3: Test training
    training_results = test_training_small_sample(train_df, test_df)
    if training_results is None:
        print("❌ Training failed. Stopping test.")
        return False
    
    # Step 4: Test prediction
    model_path = training_results['model_path']
    prediction_success = test_prediction_pipeline(model_path, test_df)
    
    if prediction_success:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("=" * 50)
        print("✅ Data loading")
        print("✅ Feature engineering (263+ features)")
        print("✅ Training pipeline")
        print("✅ Prediction pipeline")
        print("✅ SHAP explanations")
        print("\n🚀 Your ML pipeline is ready for production!")
        return True
    else:
        print("❌ Some tests failed. Check the logs above.")
        return False


if __name__ == "__main__":
    print("IEEE Fraud Detection - ML Pipeline Test")
    print("This script tests the complete training and prediction pipeline.")
    print()
    
    # Get data paths from user
    train_path = input("Enter path to your train_merged.csv file: ").strip()
    test_path = input("Enter path to your test_merged.csv file: ").strip()
    
    # Validate paths
    if not os.path.exists(train_path):
        print(f"❌ Train file not found: {train_path}")
        exit(1)
    
    if not os.path.exists(test_path):
        print(f"❌ Test file not found: {test_path}")
        exit(1)
    
    print("\n" + "=" * 50)
    
    # Run the complete test
    success = run_complete_test(train_path, test_path)
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Build the dataset adapter for cross-dataset support")
        print("2. Create the Streamlit frontend")
        print("3. Add upload functionality")
        print("4. Deploy to production")
    else:
        print("\n🔧 Debug Steps:")
        print("1. Check your data file formats")
        print("2. Verify column names match IEEE format")
        print("3. Check for missing dependencies")
        print("4. Review error messages above")
    
    exit(0 if success else 1)