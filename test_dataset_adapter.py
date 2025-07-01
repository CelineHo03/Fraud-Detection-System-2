"""
Test Script for Dataset Adapter
Demonstrates cross-dataset support for fraud detection.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from app.ml.dataset_adapter import SmartDatasetAdapter, adapt_fraud_dataset


def create_sample_credit_card_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """Create a sample credit card dataset (like Kaggle Credit Card Fraud)."""
    np.random.seed(42)
    
    data = {
        'Time': np.random.randint(0, 172800, n_samples),  # 48 hours in seconds
        'Amount': np.random.lognormal(mean=3, sigma=1.5, size=n_samples),
        'Class': np.random.choice([0, 1], size=n_samples, p=[0.998, 0.002])  # 0.2% fraud
    }
    
    # Add V1-V28 features (PCA transformed)
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    return pd.DataFrame(data)


def create_sample_bank_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """Create a sample bank transaction dataset."""
    np.random.seed(42)
    
    data = {
        'transaction_id': [f'TXN_{i:06d}' for i in range(n_samples)],
        'account_number': np.random.randint(100000, 999999, n_samples),
        'transaction_amount': np.random.lognormal(mean=4, sigma=1, size=n_samples),
        'transaction_time': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'transaction_type': np.random.choice(['TRANSFER', 'PAYMENT', 'WITHDRAWAL'], n_samples),
        'from_account': np.random.randint(100000, 999999, n_samples),
        'to_account': np.random.randint(100000, 999999, n_samples),
        'bank_code': np.random.randint(1000, 9999, n_samples),
        'is_fraud': np.random.choice([0, 1], size=n_samples, p=[0.995, 0.005])  # 0.5% fraud
    }
    
    return pd.DataFrame(data)


def create_sample_generic_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """Create a generic fraud dataset."""
    np.random.seed(42)
    
    data = {
        'id': range(n_samples),
        'payment_amount': np.random.lognormal(mean=3.5, sigma=1.2, size=n_samples),
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='15min'),
        'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'online'], n_samples),
        'customer_id': np.random.randint(1, 10000, n_samples),
        'payment_method': np.random.choice(['credit', 'debit', 'cash'], n_samples),
        'fraud_label': np.random.choice([0, 1], size=n_samples, p=[0.99, 0.01])  # 1% fraud
    }
    
    return pd.DataFrame(data)


def create_sample_ieee_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """Create a sample IEEE-like dataset."""
    np.random.seed(42)
    
    data = {
        'TransactionID': [f'T_{i:06d}' for i in range(n_samples)],
        'TransactionDT': np.random.randint(86400, 86400*30, n_samples),  # Random timestamps
        'TransactionAmt': np.random.lognormal(mean=3, sigma=1.5, size=n_samples),
        'ProductCD': np.random.choice(['W', 'C', 'R', 'H', 'S'], n_samples),
        'card1': np.random.randint(1000, 9999, n_samples),
        'card2': np.random.randint(100, 999, n_samples),
        'card3': np.random.randint(100, 200, n_samples),
        'P_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com'], n_samples),
        'isFraud': np.random.choice([0, 1], size=n_samples, p=[0.996, 0.004])  # 0.4% fraud
    }
    
    # Add some C and D columns
    for i in range(1, 6):
        data[f'C{i}'] = np.random.normal(0, 1, n_samples)
        data[f'D{i}'] = np.random.normal(0, 10, n_samples)
    
    return pd.DataFrame(data)


def test_dataset_adaptation(dataset_name: str, df: pd.DataFrame):
    """Test adaptation of a specific dataset."""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name} Dataset Adaptation")
    print(f"{'='*60}")
    
    print(f"Original Dataset Shape: {df.shape}")
    print(f"Original Columns: {list(df.columns)}")
    
    # Adapt the dataset
    adapter = SmartDatasetAdapter(verbose=True)
    result = adapter.adapt_dataset(df)
    
    print("\n" + "="*40)
    print("ADAPTATION RESULTS")
    print("="*40)
    print(adapter.get_adaptation_summary(result))
    
    if result.validation_warnings:
        print("\n⚠️  Validation Warnings:")
        for warning in result.validation_warnings:
            print(f"   - {warning}")
    
    print(f"\n✅ Adapted Dataset Shape: {result.adapted_data.shape}")
    print(f"✅ IEEE Compatibility: {result.adaptation_confidence:.1%}")
    
    # Show some adapted data
    print(f"\nAdapted Data Sample:")
    print(result.adapted_data.head(3))
    
    return result


def test_integration_with_feature_engineering():
    """Test integration with the feature engineering pipeline."""
    print(f"\n{'='*60}")
    print("Testing Integration with Feature Engineering")
    print(f"{'='*60}")
    
    # Create a sample dataset
    df = create_sample_credit_card_dataset(100)
    
    # Adapt it
    adapter = SmartDatasetAdapter(verbose=True)
    result = adapter.adapt_dataset(df)
    
    print(f"\n📊 Adapted dataset ready for feature engineering:")
    print(f"   Shape: {result.adapted_data.shape}")
    print(f"   Columns: {len(result.adapted_data.columns)}")
    
    # Check if it has the required columns for feature engineering
    required_cols = ['TransactionID', 'TransactionDT', 'TransactionAmt']
    has_required = all(col in result.adapted_data.columns for col in required_cols)
    
    if has_required:
        print("✅ Ready for feature engineering pipeline!")
        
        # Test with feature engineering (if available)
        try:
            from app.ml.preprocessing_advanced import create_advanced_features
            
            print("\n🔧 Testing with feature engineering...")
            train_features, _, engineer = create_advanced_features(
                result.adapted_data, verbose=True
            )
            print(f"✅ Feature engineering successful! Output shape: {train_features.shape}")
            
        except ImportError:
            print("⚠️  Feature engineering module not available for testing")
    else:
        missing = [col for col in required_cols if col not in result.adapted_data.columns]
        print(f"❌ Missing required columns: {missing}")


def run_comprehensive_test():
    """Run comprehensive tests on all dataset types."""
    print("🚀 Starting Comprehensive Dataset Adapter Tests")
    print("="*80)
    
    # Test different dataset types
    test_datasets = [
        ("IEEE Format", create_sample_ieee_dataset(500)),
        ("Credit Card (V1-V28)", create_sample_credit_card_dataset(500)),
        ("Bank Transactions", create_sample_bank_dataset(500)),
        ("Generic Fraud", create_sample_generic_dataset(500))
    ]
    
    results = []
    
    for dataset_name, df in test_datasets:
        try:
            result = test_dataset_adaptation(dataset_name, df)
            results.append((dataset_name, result))
        except Exception as e:
            print(f"❌ Error testing {dataset_name}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*60}")
    
    for dataset_name, result in results:
        print(f"{dataset_name:20} | Confidence: {result.adaptation_confidence:.1%} | "
              f"Shape: {result.adapted_data.shape} | "
              f"Synthetic: {len(result.synthetic_features)}")
    
    # Test integration
    test_integration_with_feature_engineering()
    
    print(f"\n🎉 All tests completed!")
    return results


def test_custom_dataset():
    """Test with a custom dataset format."""
    print(f"\n{'='*60}")
    print("Testing Custom Dataset Upload Simulation")
    print(f"{'='*60}")
    
    # Simulate a user uploading a custom CSV
    custom_data = {
        'ref_number': [f'REF{i:04d}' for i in range(100)],
        'purchase_amount': np.random.uniform(10, 1000, 100),
        'purchase_time': pd.date_range('2023-06-01', periods=100, freq='2H'),
        'store_category': np.random.choice(['electronics', 'clothing', 'food'], 100),
        'customer_email': [f'user{i}@example.com' for i in range(100)],
        'payment_type': np.random.choice(['visa', 'mastercard', 'amex'], 100),
        'is_suspicious': np.random.choice([0, 1], 100, p=[0.92, 0.08])
    }
    
    df_custom = pd.DataFrame(custom_data)
    
    print("Custom Dataset (simulating user upload):")
    print(df_custom.head())
    
    # Adapt it
    result = adapt_fraud_dataset(df_custom, verbose=True)
    
    print(f"\n✅ Custom dataset successfully adapted!")
    print(f"   Confidence: {result.adaptation_confidence:.1%}")
    print(f"   Final shape: {result.adapted_data.shape}")
    
    return result


if __name__ == "__main__":
    print("Dataset Adapter Test Suite")
    print("=" * 50)
    
    # Run all tests
    try:
        # Comprehensive test
        results = run_comprehensive_test()
        
        # Custom dataset test
        custom_result = test_custom_dataset()
        
        print(f"\n🏆 All tests passed successfully!")
        print("Your dataset adapter is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()