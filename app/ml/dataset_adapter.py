"""
Cross-Dataset Adapter for Fraud Detection
Intelligently adapts any fraud dataset to IEEE format for prediction.

This module provides the core functionality to:
- Auto-detect dataset formats (IEEE, Credit Card, Bank, Generic)
- Map columns intelligently using fuzzy matching and heuristics
- Create synthetic IEEE features where missing
- Validate and clean data for prediction
- Ensure compatibility with the trained IEEE model
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DatasetType(Enum):
    """Supported dataset types for fraud detection."""
    IEEE = "ieee"
    CREDIT_CARD = "credit_card"
    BANK_TRANSACTION = "bank_transaction"
    GENERIC_FRAUD = "generic_fraud"
    CUSTOM = "custom"


@dataclass
class ColumnMapping:
    """Store column mapping information."""
    original_name: str
    mapped_name: str
    confidence: float
    mapping_type: str  # 'exact', 'fuzzy', 'semantic', 'synthetic'


@dataclass
class AdaptationResult:
    """Result of dataset adaptation."""
    adapted_data: pd.DataFrame
    dataset_type: DatasetType
    column_mappings: List[ColumnMapping]
    synthetic_features: List[str]
    validation_warnings: List[str]
    adaptation_confidence: float


class SmartDatasetAdapter:
    """
    Intelligent dataset adapter that can transform any fraud dataset 
    into IEEE format for prediction.
    """
    
    # IEEE column specifications
    IEEE_REQUIRED_COLUMNS = [
        'TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD'
    ]
    
    IEEE_OPTIONAL_COLUMNS = [
        'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'dist1', 'dist2',
        'P_emaildomain', 'R_emaildomain',
        'DeviceType', 'DeviceInfo'
    ]
    
    # Column mapping patterns
    COLUMN_PATTERNS = {
        # Transaction identifiers
        'TransactionID': [
            'transaction_id', 'transactionid', 'trans_id', 'id', 'transaction_number',
            'reference', 'ref_no', 'payment_id', 'order_id'
        ],
        
        # Transaction amount
        'TransactionAmt': [
            'amount', 'transaction_amount', 'trans_amount', 'payment_amount',
            'value', 'sum', 'total', 'price', 'cost', 'charge'
        ],
        
        # Transaction time
        'TransactionDT': [
            'timestamp', 'time', 'date', 'transaction_time', 'trans_time',
            'payment_time', 'created_at', 'datetime', 'transaction_date'
        ],
        
        # Product/Category
        'ProductCD': [
            'product', 'category', 'product_code', 'type', 'class',
            'merchant_category', 'mcc', 'product_type'
        ],
        
        # Card information
        'card1': ['card_number', 'card_id', 'card', 'pan', 'card_1'],
        'card2': ['card_type', 'card_brand', 'issuer', 'card_2'],
        'card3': ['card_family', 'card_category', 'card_3'],
        'card4': ['card_feature', 'card_4'],
        'card5': ['card_issue_bank', 'bank', 'card_5'],
        'card6': ['card_type_2', 'card_6'],
        
        # Address information
        'addr1': ['address', 'addr', 'billing_address', 'customer_address'],
        'addr2': ['shipping_address', 'delivery_address', 'addr_2'],
        
        # Distance features
        'dist1': ['distance', 'dist', 'location_distance'],
        'dist2': ['distance_2', 'dist_2'],
        
        # Email domains
        'P_emaildomain': ['email_domain', 'email', 'purchaser_email'],
        'R_emaildomain': ['recipient_email', 'receiver_email'],
        
        # Device information
        'DeviceType': ['device_type', 'device', 'platform'],
        'DeviceInfo': ['device_info', 'user_agent', 'browser']
    }
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the dataset adapter.
        
        Args:
            verbose: Whether to print adaptation progress
        """
        self.verbose = verbose
        self.adaptation_history = []
    
    def _safe_pandas_sum(self, series_result) -> int:
        """
        Safely convert pandas sum results to integer.
        
        Args:
            series_result: Result from pandas sum operation
            
        Returns:
            Integer value
        """
        try:
            # Handle different pandas return types
            if hasattr(series_result, 'iloc'):
                # It's a Series, get the first value
                return int(series_result.iloc[0]) if len(series_result) > 0 else 0
            elif hasattr(series_result, 'item'):
                # It's a numpy scalar
                return int(series_result.item())
            else:
                # It's already a scalar
                return int(series_result)
        except (TypeError, ValueError, AttributeError):
            # If all else fails, return 0
            return 0
    
    def detect_dataset_type(self, df: pd.DataFrame) -> DatasetType:
        """
        Automatically detect the type of fraud dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Detected dataset type
        """
        if self.verbose:
            print("Detecting dataset type...")
        
        columns_lower = [col.lower() for col in df.columns]
        
        # Check for IEEE format
        ieee_indicators = [
            'transactionid', 'transactiondt', 'transactionamt', 'productcd'
        ]
        ieee_matches = sum(1 for indicator in ieee_indicators if indicator in columns_lower)
        
        if ieee_matches >= 3:
            return DatasetType.IEEE
        
        # Check for credit card format (V1-V28 features)
        v_features = [col for col in df.columns if re.match(r'^V\d+$', col)]
        if len(v_features) >= 10:
            return DatasetType.CREDIT_CARD
        
        # Check for bank transaction format
        bank_indicators = [
            'account', 'balance', 'bank', 'branch', 'iban', 'swift'
        ]
        bank_matches = sum(1 for indicator in bank_indicators 
                          if any(indicator in col.lower() for col in df.columns))
        
        if bank_matches >= 2:
            return DatasetType.BANK_TRANSACTION
        
        # Check for generic fraud indicators
        fraud_indicators = [
            'fraud', 'label', 'target', 'class', 'is_fraud'
        ]
        has_fraud_label = any(indicator in col.lower() for col in df.columns 
                             for indicator in fraud_indicators)
        
        transaction_indicators = ['amount', 'time', 'date', 'id']
        transaction_matches = sum(1 for indicator in transaction_indicators 
                                if any(indicator in col.lower() for col in df.columns))
        
        if has_fraud_label and transaction_matches >= 2:
            return DatasetType.GENERIC_FRAUD
        
        return DatasetType.CUSTOM
    
    def map_columns(self, df: pd.DataFrame, dataset_type: DatasetType) -> List[ColumnMapping]:
        """
        Intelligently map columns to IEEE format.
        
        Args:
            df: Input DataFrame
            dataset_type: Detected dataset type
            
        Returns:
            List of column mappings
        """
        if self.verbose:
            print(f"Mapping columns for {dataset_type.value} format...")
        
        mappings = []
        
        # Handle different dataset types
        if dataset_type == DatasetType.IEEE:
            mappings = self._map_ieee_columns(df)
        elif dataset_type == DatasetType.CREDIT_CARD:
            mappings = self._map_credit_card_columns(df)
        elif dataset_type == DatasetType.BANK_TRANSACTION:
            mappings = self._map_bank_columns(df)
        else:
            mappings = self._map_generic_columns(df)
        
        return mappings
    
    def _map_ieee_columns(self, df: pd.DataFrame) -> List[ColumnMapping]:
        """Map IEEE format columns (mostly direct mapping)."""
        mappings = []
        
        for col in df.columns:
            # Direct mapping for IEEE columns
            confidence = 1.0 if col in self.IEEE_REQUIRED_COLUMNS + self.IEEE_OPTIONAL_COLUMNS else 0.8
            mappings.append(ColumnMapping(
                original_name=col,
                mapped_name=col,
                confidence=confidence,
                mapping_type='exact'
            ))
        
        return mappings
    
    def _map_credit_card_columns(self, df: pd.DataFrame) -> List[ColumnMapping]:
        """Map credit card dataset columns (V1-V28 format)."""
        mappings = []
        
        # Map V features directly
        for col in df.columns:
            if re.match(r'^V\d+$', col):
                mappings.append(ColumnMapping(
                    original_name=col,
                    mapped_name=col,
                    confidence=1.0,
                    mapping_type='exact'
                ))
            else:
                # Use fuzzy matching for other columns
                mapped_name, confidence = self._fuzzy_match_column(col)
                if mapped_name:
                    mappings.append(ColumnMapping(
                        original_name=col,
                        mapped_name=mapped_name,
                        confidence=confidence,
                        mapping_type='fuzzy'
                    ))
        
        return mappings
    
    def _map_bank_columns(self, df: pd.DataFrame) -> List[ColumnMapping]:
        """Map bank transaction dataset columns."""
        mappings = []
        
        # Special handling for bank-specific columns
        bank_mappings = {
            'account_number': 'card1',
            'account_id': 'card1',
            'from_account': 'card1',
            'to_account': 'card2',
            'bank_code': 'card3',
            'branch_code': 'card4',
            'transaction_code': 'ProductCD'
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check bank-specific mappings first
            if col_lower in bank_mappings:
                mappings.append(ColumnMapping(
                    original_name=col,
                    mapped_name=bank_mappings[col_lower],
                    confidence=0.9,
                    mapping_type='semantic'
                ))
            else:
                # Use fuzzy matching
                mapped_name, confidence = self._fuzzy_match_column(col)
                if mapped_name:
                    mappings.append(ColumnMapping(
                        original_name=col,
                        mapped_name=mapped_name,
                        confidence=confidence,
                        mapping_type='fuzzy'
                    ))
        
        return mappings
    
    def _map_generic_columns(self, df: pd.DataFrame) -> List[ColumnMapping]:
        """Map generic fraud dataset columns using fuzzy matching."""
        mappings = []
        
        for col in df.columns:
            mapped_name, confidence = self._fuzzy_match_column(col)
            if mapped_name:
                mappings.append(ColumnMapping(
                    original_name=col,
                    mapped_name=mapped_name,
                    confidence=confidence,
                    mapping_type='fuzzy'
                ))
        
        return mappings
    
    def _fuzzy_match_column(self, column_name: str) -> Tuple[Optional[str], float]:
        """
        Fuzzy match a column name to IEEE format.
        
        Args:
            column_name: Original column name
            
        Returns:
            Tuple of (mapped_name, confidence_score)
        """
        col_lower = column_name.lower().strip()
        
        # Remove common prefixes/suffixes and special characters
        cleaned_col = re.sub(r'[^a-z0-9_]', '', col_lower)
        
        best_match = None
        best_score = 0.0
        
        for ieee_col, patterns in self.COLUMN_PATTERNS.items():
            for pattern in patterns:
                # Exact match
                if cleaned_col == pattern.replace(' ', '_'):
                    return ieee_col, 1.0
                
                # Substring match
                if pattern in cleaned_col or cleaned_col in pattern:
                    score = len(pattern) / max(len(cleaned_col), len(pattern))
                    if score > best_score:
                        best_match = ieee_col
                        best_score = score
                
                # Keyword match
                keywords = pattern.split('_')
                if all(keyword in cleaned_col for keyword in keywords):
                    score = 0.8
                    if score > best_score:
                        best_match = ieee_col
                        best_score = score
        
        # Only return matches with reasonable confidence
        if best_score >= 0.6:
            return best_match, best_score
        
        return None, 0.0
    
    def create_synthetic_features(self, df: pd.DataFrame, mappings: List[ColumnMapping]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create synthetic IEEE features for missing columns.
        
        Args:
            df: DataFrame with mapped columns
            mappings: Column mappings
            
        Returns:
            Tuple of (enhanced_dataframe, list_of_synthetic_features)
        """
        if self.verbose:
            print("Creating synthetic features...")
        
        df_enhanced = df.copy()
        synthetic_features = []
        
        # Get mapped column names
        mapped_columns = {mapping.mapped_name for mapping in mappings}
        
        # Create missing required columns
        for required_col in self.IEEE_REQUIRED_COLUMNS:
            if required_col not in mapped_columns:
                df_enhanced[required_col] = self._synthesize_column(df_enhanced, required_col)
                synthetic_features.append(required_col)
        
        # Create some optional columns that are important for the model
        important_optional = ['card1', 'card2', 'ProductCD']
        for opt_col in important_optional:
            if opt_col not in mapped_columns:
                df_enhanced[opt_col] = self._synthesize_column(df_enhanced, opt_col)
                synthetic_features.append(opt_col)
        
        # Create C and D columns (IEEE competition features) as zeros
        # These will be filled by feature engineering if needed
        for i in range(1, 15):
            col_c = f'C{i}'
            col_d = f'D{i}'
            if col_c not in mapped_columns:
                df_enhanced[col_c] = 0.0
                synthetic_features.append(col_c)
            if col_d not in mapped_columns:
                df_enhanced[col_d] = np.nan  # D columns can be NaN
                synthetic_features.append(col_d)
        
        # Create M columns (match status) as 'T' (True) by default
        for i in range(1, 10):
            col_m = f'M{i}'
            if col_m not in mapped_columns:
                df_enhanced[col_m] = 'T'
                synthetic_features.append(col_m)
        
        return df_enhanced, synthetic_features
    
    def _synthesize_column(self, df: pd.DataFrame, column_name: str) -> pd.Series:
        """
        Synthesize a specific IEEE column based on available data.
        
        Args:
            df: Input DataFrame
            column_name: IEEE column to synthesize
            
        Returns:
            Synthetic column data
        """
        n_rows = len(df)
        
        if column_name == 'TransactionID':
            # Create unique transaction IDs
            return pd.Series([f'T_{i:06d}' for i in range(n_rows)], index=df.index)
        
        elif column_name == 'TransactionDT':
            # Create timestamp (seconds since epoch)
            base_time = datetime(2019, 1, 1).timestamp()
            return pd.Series(np.arange(base_time, base_time + n_rows * 3600, 3600), index=df.index)
        
        elif column_name == 'TransactionAmt':
            # Create reasonable transaction amounts
            return pd.Series(np.random.lognormal(mean=3, sigma=1, size=n_rows), index=df.index)
        
        elif column_name == 'ProductCD':
            # Create product codes
            products = ['W', 'C', 'R', 'H', 'S']
            return pd.Series(np.random.choice(products, size=n_rows), index=df.index)
        
        elif column_name.startswith('card'):
            # Create card information
            if column_name == 'card1':
                return pd.Series(np.random.randint(1000, 9999, size=n_rows), index=df.index)
            else:
                return pd.Series(np.random.randint(100, 999, size=n_rows), index=df.index)
        
        else:
            # Default: create zeros or random values
            return pd.Series(np.zeros(n_rows), index=df.index)
    
    def validate_and_clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate and clean the adapted dataset.
        
        Args:
            df: Adapted DataFrame
            
        Returns:
            Tuple of (cleaned_dataframe, validation_warnings)
        """
        if self.verbose:
            print("Validating and cleaning data...")
        
        df_clean = df.copy()
        warnings = []
        
        # Remove duplicate columns that might cause pandas issues
        if df_clean.columns.duplicated().any():
            duplicate_cols = df_clean.columns[df_clean.columns.duplicated()].tolist()
            warnings.append(f"Removed duplicate columns: {duplicate_cols}")
            df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]
        
        # Check for required columns
        missing_required = [col for col in self.IEEE_REQUIRED_COLUMNS if col not in df_clean.columns]
        if missing_required:
            warnings.append(f"Missing required columns: {missing_required}")
        
        # Validate data types
        if 'TransactionAmt' in df_clean.columns:
            # Ensure TransactionAmt is numeric and positive
            df_clean['TransactionAmt'] = pd.to_numeric(df_clean['TransactionAmt'], errors='coerce')
            try:
                negative_amounts = self._safe_pandas_sum((df_clean['TransactionAmt'] < 0).sum())
                if negative_amounts > 0:
                    warnings.append(f"Found {negative_amounts} negative transaction amounts")
                    df_clean['TransactionAmt'] = df_clean['TransactionAmt'].abs()
            except Exception as e:
                warnings.append(f"Could not validate TransactionAmt: {e}")
        
        if 'TransactionDT' in df_clean.columns:
            # Ensure TransactionDT is numeric
            df_clean['TransactionDT'] = pd.to_numeric(df_clean['TransactionDT'], errors='coerce')
        
        # Handle missing values
        try:
            # Get column types more safely
            all_columns = df_clean.columns.tolist()
            numeric_columns = []
            categorical_columns = []
            
            for col in all_columns:
                try:
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        numeric_columns.append(col)
                    elif pd.api.types.is_object_dtype(df_clean[col]) or pd.api.types.is_categorical_dtype(df_clean[col]):
                        categorical_columns.append(col)
                    # Skip datetime columns for now
                except Exception:
                    # If we can't determine type, treat as categorical
                    categorical_columns.append(col)
                    
        except Exception as e:
            warnings.append(f"Could not determine column types: {e}")
            return df_clean, warnings
        
        # Fill missing values
        for col in numeric_columns:
            try:
                missing_count = self._safe_pandas_sum(df_clean[col].isnull().sum())
                if missing_count > 0:
                    if col.startswith('C') or col.startswith('V'):
                        df_clean[col] = df_clean[col].fillna(0)
                    else:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    warnings.append(f"Filled {missing_count} missing values in {col}")
            except Exception as e:
                warnings.append(f"Could not process numeric column {col}: {e}")
                continue
        
        for col in categorical_columns:
            try:
                missing_count = self._safe_pandas_sum(df_clean[col].isnull().sum())
                if missing_count > 0:
                    df_clean[col] = df_clean[col].fillna('unknown')
                    warnings.append(f"Filled {missing_count} missing values in {col}")
            except Exception as e:
                warnings.append(f"Could not process categorical column {col}: {e}")
                continue
        
        return df_clean, warnings
    
    def adapt_dataset(self, df: pd.DataFrame) -> AdaptationResult:
        """
        Main method to adapt any dataset to IEEE format.
        
        Args:
            df: Input fraud dataset
            
        Returns:
            AdaptationResult with adapted data and metadata
        """
        if self.verbose:
            print("=== Starting Dataset Adaptation ===")
            print(f"Input shape: {df.shape}")
        
        # Step 1: Detect dataset type
        dataset_type = self.detect_dataset_type(df)
        if self.verbose:
            print(f"Detected dataset type: {dataset_type.value}")
        
        # Step 2: Map columns
        column_mappings = self.map_columns(df, dataset_type)
        if self.verbose:
            print(f"Mapped {len(column_mappings)} columns")
        
        # Step 3: Apply column mappings
        df_mapped = df.copy()
        for mapping in column_mappings:
            if mapping.mapped_name != mapping.original_name:
                df_mapped = df_mapped.rename(columns={mapping.original_name: mapping.mapped_name})
        
        # Step 4: Create synthetic features
        df_enhanced, synthetic_features = self.create_synthetic_features(df_mapped, column_mappings)
        if self.verbose:
            print(f"Created {len(synthetic_features)} synthetic features")
        
        # Step 5: Validate and clean
        df_final, validation_warnings = self.validate_and_clean(df_enhanced)
        
        # Step 6: Calculate adaptation confidence
        total_mappings = len(column_mappings)
        high_confidence_mappings = sum(1 for m in column_mappings if m.confidence >= 0.8)
        adaptation_confidence = high_confidence_mappings / total_mappings if total_mappings > 0 else 0.0
        
        if self.verbose:
            print(f"Final shape: {df_final.shape}")
            print(f"Adaptation confidence: {adaptation_confidence:.2f}")
            print("=== Dataset Adaptation Complete ===")
        
        # Store adaptation history
        self.adaptation_history.append({
            'timestamp': datetime.now(),
            'input_shape': df.shape,
            'output_shape': df_final.shape,
            'dataset_type': dataset_type,
            'confidence': adaptation_confidence
        })
        
        return AdaptationResult(
            adapted_data=df_final,
            dataset_type=dataset_type,
            column_mappings=column_mappings,
            synthetic_features=synthetic_features,
            validation_warnings=validation_warnings,
            adaptation_confidence=adaptation_confidence
        )
    
    def get_adaptation_summary(self, result: AdaptationResult) -> str:
        """
        Generate a human-readable summary of the adaptation process.
        
        Args:
            result: AdaptationResult from adapt_dataset
            
        Returns:
            Formatted summary string
        """
        summary = []
        summary.append(f"Dataset Type: {result.dataset_type.value}")
        summary.append(f"Adaptation Confidence: {result.adaptation_confidence:.1%}")
        summary.append(f"Final Dataset Shape: {result.adapted_data.shape}")
        
        # Column mappings summary
        exact_mappings = sum(1 for m in result.column_mappings if m.mapping_type == 'exact')
        fuzzy_mappings = sum(1 for m in result.column_mappings if m.mapping_type == 'fuzzy')
        semantic_mappings = sum(1 for m in result.column_mappings if m.mapping_type == 'semantic')
        
        summary.append(f"Column Mappings: {exact_mappings} exact, {fuzzy_mappings} fuzzy, {semantic_mappings} semantic")
        summary.append(f"Synthetic Features: {len(result.synthetic_features)}")
        
        if result.validation_warnings:
            summary.append(f"Warnings: {len(result.validation_warnings)}")
        
        return "\n".join(summary)


# Convenience function for easy usage
def adapt_fraud_dataset(df: pd.DataFrame, verbose: bool = True) -> AdaptationResult:
    """
    Convenience function to adapt any fraud dataset to IEEE format.
    
    Args:
        df: Input fraud dataset
        verbose: Whether to print progress
        
    Returns:
        AdaptationResult with adapted data
    """
    adapter = SmartDatasetAdapter(verbose=verbose)
    return adapter.adapt_dataset(df)


# Example usage and testing
if __name__ == "__main__":
    print("Smart Dataset Adapter for Fraud Detection")
    print("Supports: IEEE, Credit Card, Bank Transaction, Generic Fraud datasets")
    print("Use adapt_fraud_dataset(df) to adapt your dataset to IEEE format.")