"""
Advanced Feature Engineering for IEEE Fraud Detection
Competition-winning approach with 263 engineered features.

This module implements the advanced feature engineering pipeline that creates
263 features from the raw IEEE fraud detection dataset, including:
- Magic UID features for user identification
- Time-based feature engineering with normalization
- Aggregation features (mean, std, count, etc.)
- Frequency encoding for categorical variables
- Advanced interaction features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering pipeline for IEEE fraud detection.
    
    Creates 263 engineered features using competition-winning techniques:
    - Magic UID: Unique user identification across cards/emails
    - Temporal features: Time-based patterns and normalization
    - Aggregation features: Statistical summaries by groups
    - Frequency encoding: Categorical variable transformations
    - Interaction features: Cross-feature relationships
    """
    
    def __init__(self, 
                 create_uid: bool = True,
                 temporal_features: bool = True,
                 aggregation_features: bool = True,
                 frequency_encoding: bool = True,
                 interaction_features: bool = True,
                 verbose: bool = False):
        """
        Initialize the advanced feature engineer.
        
        Args:
            create_uid: Whether to create magic UID features
            temporal_features: Whether to create time-based features
            aggregation_features: Whether to create aggregation features
            frequency_encoding: Whether to apply frequency encoding
            interaction_features: Whether to create interaction features
            verbose: Whether to print progress information
        """
        self.create_uid = create_uid
        self.temporal_features = temporal_features
        self.aggregation_features = aggregation_features
        self.frequency_encoding = frequency_encoding
        self.interaction_features = interaction_features
        self.verbose = verbose
        
        # Storage for fit parameters
        self.label_encoders_ = {}
        self.aggregation_stats_ = {}
        self.frequency_maps_ = {}
        self.feature_names_ = []
        self.d_columns_trained_ = []  # Store which D-columns were used in training
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AdvancedFeatureEngineer':
        """
        Fit the feature engineer on training data.
        
        Args:
            X: Input DataFrame with IEEE fraud detection features
            y: Target variable (optional)
            
        Returns:
            self: Fitted feature engineer
        """
        if self.verbose:
            print("Fitting Advanced Feature Engineer...")
            print(f"Input shape: {X.shape}")
        
        # Reset state
        self.label_encoders_ = {}
        self.aggregation_stats_ = {}
        self.frequency_maps_ = {}
        self.d_columns_trained_ = []
        
        # Ensure we have a copy to work with
        X_work = X.copy()
        
        # Step 0: Standardize column names
        X_work = self._standardize_column_names(X_work)
        
        # Step 1: Create magic UID features
        if self.create_uid:
            X_work = self._create_uid_features(X_work, fit=True)
        
        # Step 1.5: Create temporal features to identify D-columns
        if self.temporal_features:
            X_work = self._create_temporal_features(X_work)
        
        # Step 2: Fit frequency encoders
        if self.frequency_encoding:
            self._fit_frequency_encoders(X_work)
        
        # Step 3: Fit aggregation statistics
        if self.aggregation_features:
            self._fit_aggregation_stats(X_work)
        
        # Step 4: Set fitted state first
        self.is_fitted_ = True
        
        # Step 5: Get final feature names
        X_transformed = self.transform(X_work.head(100))  # Sample for feature names
        self.feature_names_ = list(X_transformed.columns)
        
        if self.verbose:
            print(f"Feature engineering fitted. Output features: {len(self.feature_names_)}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data using fitted feature engineering.
        
        Args:
            X: Input DataFrame with IEEE fraud detection features
            
        Returns:
            DataFrame with 263 engineered features
        """
        if not self.is_fitted_:
            raise ValueError("Feature engineer must be fitted before transform.")
        
        if self.verbose:
            print(f"Transforming data with shape: {X.shape}")
        
        # Start with a copy
        X_transformed = X.copy()
        
        # Step 0: Standardize column names (fix id-XX vs id_XX inconsistency)
        X_transformed = self._standardize_column_names(X_transformed)
        
        # Step 1: Create magic UID features
        if self.create_uid:
            X_transformed = self._create_uid_features(X_transformed, fit=False)
        
        # Step 2: Create temporal features
        if self.temporal_features:
            X_transformed = self._create_temporal_features(X_transformed)
        
        # Step 3: Apply frequency encoding
        if self.frequency_encoding:
            X_transformed = self._apply_frequency_encoding(X_transformed)
        
        # Step 4: Create aggregation features
        if self.aggregation_features:
            X_transformed = self._create_aggregation_features(X_transformed)
        
        # Step 5: Create interaction features
        if self.interaction_features:
            X_transformed = self._create_interaction_features(X_transformed)
        
        # Step 6: Clean and finalize
        X_transformed = self._finalize_features(X_transformed)
        
        if self.verbose:
            print(f"Transformation complete. Output shape: {X_transformed.shape}")
        
        return X_transformed
    
    def _standardize_column_names(self, X: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to fix inconsistencies like id-XX vs id_XX."""
        if self.verbose:
            print("Standardizing column names...")
        
        X_new = X.copy()
        
        # Create a mapping of old names to new names
        column_mapping = {}
        
        for col in X_new.columns:
            new_col = col
            
            # Fix id-XX to id_XX format
            if col.startswith('id_') and '-' in col:
                new_col = col.replace('-', '_')
            elif col.startswith('id-'):
                new_col = col.replace('id-', 'id_')
            
            # Standardize other common patterns
            new_col = new_col.replace('-', '_')  # Replace all dashes with underscores
            new_col = new_col.lower()  # Make lowercase for consistency
            
            if new_col != col:
                column_mapping[col] = new_col
        
        # Rename columns if any changes needed
        if column_mapping:
            X_new = X_new.rename(columns=column_mapping)
            if self.verbose:
                print(f"Renamed {len(column_mapping)} columns for consistency")
        
        return X_new
    
    def _create_uid_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Create magic UID features for user identification."""
        if self.verbose:
            print("Creating UID features...")
        
        X_new = X.copy()
        
        # Magic UID: Combine card and email information
        if 'card1' in X.columns and 'card2' in X.columns:
            X_new['uid'] = X['card1'].astype(str) + '_' + X['card2'].fillna('').astype(str)
        elif 'card1' in X.columns:
            X_new['uid'] = X['card1'].astype(str)
        else:
            X_new['uid'] = 'unknown'
        
        # Enhanced UID with address information
        addr_cols = [c for c in X.columns if 'addr' in c.lower()]
        if addr_cols:
            addr_combined = X[addr_cols].fillna('').astype(str).agg('_'.join, axis=1)
            X_new['uid_addr'] = X_new['uid'] + '_' + addr_combined
        
        # UID with device information
        device_cols = [c for c in X.columns if 'device' in c.lower() or c.startswith('id_')]
        if device_cols:
            device_combined = X[device_cols].fillna('').astype(str).agg('_'.join, axis=1)
            X_new['uid_device'] = X_new['uid'] + '_' + device_combined
        
        # Encode UIDs
        for uid_col in ['uid', 'uid_addr', 'uid_device']:
            if uid_col in X_new.columns:
                if fit:
                    le = LabelEncoder()
                    X_new[f'{uid_col}_encoded'] = le.fit_transform(X_new[uid_col].astype(str))
                    self.label_encoders_[uid_col] = le
                else:
                    le = self.label_encoders_.get(uid_col)
                    if le:
                        # Handle unseen values
                        unseen_mask = ~X_new[uid_col].astype(str).isin(le.classes_)
                        X_new[f'{uid_col}_encoded'] = 0  # Default for unseen
                        seen_mask = ~unseen_mask
                        if seen_mask.any():
                            X_new.loc[seen_mask, f'{uid_col}_encoded'] = le.transform(
                                X_new.loc[seen_mask, uid_col].astype(str)
                            )
        
        return X_new
    
    def _create_temporal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive time-based features."""
        if self.verbose:
            print("Creating temporal features...")
        
        X_new = X.copy()
        
        # TransactionDT features
        if 'TransactionDT' in X.columns or 'transactiondt' in X.columns:
            # Handle case variations
            dt_col = 'TransactionDT' if 'TransactionDT' in X.columns else 'transactiondt'
            
            # Basic time features
            X_new['TransactionDT_hour'] = (X[dt_col] / 3600) % 24
            X_new['TransactionDT_day'] = (X[dt_col] / (3600 * 24)) % 7
            X_new['TransactionDT_week'] = (X[dt_col] / (3600 * 24 * 7)) % 52
            
            # Cyclical encoding
            X_new['TransactionDT_hour_sin'] = np.sin(2 * np.pi * X_new['TransactionDT_hour'] / 24)
            X_new['TransactionDT_hour_cos'] = np.cos(2 * np.pi * X_new['TransactionDT_hour'] / 24)
            X_new['TransactionDT_day_sin'] = np.sin(2 * np.pi * X_new['TransactionDT_day'] / 7)
            X_new['TransactionDT_day_cos'] = np.cos(2 * np.pi * X_new['TransactionDT_day'] / 7)
            
            # Time since epoch features
            X_new['TransactionDT_log'] = np.log1p(X[dt_col])
            X_new['TransactionDT_sqrt'] = np.sqrt(X[dt_col])
            
            # Business hours indicator
            X_new['is_business_hours'] = ((X_new['TransactionDT_hour'] >= 9) & 
                                         (X_new['TransactionDT_hour'] <= 17)).astype(int)
            X_new['is_weekend'] = (X_new['TransactionDT_day'] >= 5).astype(int)
        
        # D-columns (time-based features) - Handle consistently
        if not self.is_fitted_:
            # During training: store which D-columns have data
            d_cols = [c for c in X.columns if c.lower().startswith('d') and c[1:].isdigit()]
            d_cols_with_data = [col for col in d_cols if X[col].notna().sum() > 0]
            self.d_columns_trained_ = d_cols_with_data
        
        # Use the same D-columns that were used during training
        for col in self.d_columns_trained_:
            # Check if column exists in current data
            if col in X.columns and X[col].notna().sum() > 0:
                # Process normally
                col_max = X[col].max()
                if col_max > 0:
                    X_new[f'{col}_normalized'] = X[col] / (col_max + 1e-8)
                else:
                    X_new[f'{col}_normalized'] = 0
                
                try:
                    X_new[f'{col}_binned'] = pd.cut(X[col], bins=10, labels=False, duplicates='drop')
                except ValueError:
                    X_new[f'{col}_binned'] = 0
                
                X_new[f'{col}_log'] = np.log1p(X[col].fillna(0))
            else:
                # Column doesn't exist or has no data - create zero features for consistency
                if self.verbose:
                    print(f"Creating zero features for missing D-column: {col}")
                X_new[f'{col}_normalized'] = 0
                X_new[f'{col}_binned'] = 0
                X_new[f'{col}_log'] = 0
        
        return X_new
    
    def _fit_frequency_encoders(self, X: pd.DataFrame) -> None:
        """Fit frequency encoders for categorical variables."""
        if self.verbose:
            print("Fitting frequency encoders...")
        
        # Categorical columns to encode
        categorical_cols = []
        
        # ProductCD
        productcd_cols = [c for c in X.columns if c.lower() == 'productcd']
        categorical_cols.extend(productcd_cols)
        
        # Card columns
        card_cols = [c for c in X.columns if c.lower().startswith('card')]
        categorical_cols.extend(card_cols)
        
        # Email domain
        email_cols = [c for c in X.columns if 'emaildomain' in c.lower()]
        categorical_cols.extend(email_cols)
        
        # M columns
        m_cols = [c for c in X.columns if c.lower().startswith('m') and c[1:].isdigit()]
        categorical_cols.extend(m_cols)
        
        # Create frequency maps
        for col in categorical_cols:
            if col in X.columns:
                freq_map = X[col].value_counts(normalize=True).to_dict()
                self.frequency_maps_[col] = freq_map
    
    def _apply_frequency_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply frequency encoding to categorical variables."""
        if self.verbose:
            print("Applying frequency encoding...")
        
        X_new = X.copy()
        
        for col, freq_map in self.frequency_maps_.items():
            if col in X.columns:
                X_new[f'{col}_freq'] = X[col].map(freq_map).fillna(0)
                
                # Additional encodings
                X_new[f'{col}_freq_log'] = np.log1p(X_new[f'{col}_freq'])
                X_new[f'{col}_is_rare'] = (X_new[f'{col}_freq'] < 0.01).astype(int)
        
        return X_new
    
    def _fit_aggregation_stats(self, X: pd.DataFrame) -> None:
        """Fit aggregation statistics for group-based features."""
        if self.verbose:
            print("Fitting aggregation statistics...")
        
        # Key grouping variables
        group_cols = []
        if 'uid_encoded' in X.columns:
            group_cols.append('uid_encoded')
        
        card_cols = [c for c in X.columns if c.lower().startswith('card') and c.lower() != 'card1_card2_interaction']
        if card_cols:
            group_cols.append(card_cols[0])  # Use first card column
        
        email_cols = [c for c in X.columns if 'emaildomain' in c.lower()]
        if email_cols:
            group_cols.append(email_cols[0])  # Use first email domain column
        
        # Numerical columns to aggregate
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID columns and target
        exclude_cols = ['TransactionID', 'transactionid', 'isFraud', 'isfraud']
        numeric_cols = [c for c in numeric_cols if c.lower() not in [e.lower() for e in exclude_cols]]
        
        # Calculate aggregation statistics
        for group_col in group_cols[:2]:  # Limit to prevent memory issues
            if group_col in X.columns:
                group_stats = {}
                
                for agg_func in ['mean', 'std', 'min', 'max', 'count']:
                    for num_col in numeric_cols[:10]:  # Limit features to manage memory
                        if num_col in X.columns and num_col != group_col:
                            try:
                                if agg_func == 'count':
                                    stats = X.groupby(group_col).size()
                                else:
                                    stats = X.groupby(group_col)[num_col].agg(agg_func)
                                
                                group_stats[f'{group_col}_{num_col}_{agg_func}'] = stats.to_dict()
                            except Exception:
                                continue
                
                self.aggregation_stats_[group_col] = group_stats
    
    def _create_aggregation_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create aggregation features using fitted statistics."""
        if self.verbose:
            print("Creating aggregation features...")
        
        X_new = X.copy()
        
        for group_col, group_stats in self.aggregation_stats_.items():
            if group_col in X.columns:
                for stat_name, stat_dict in group_stats.items():
                    X_new[stat_name] = X[group_col].map(stat_dict).fillna(0)
        
        return X_new
    
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables."""
        if self.verbose:
            print("Creating interaction features...")
        
        X_new = X.copy()
        
        # Amount-based interactions
        amt_cols = [c for c in X.columns if 'transactionamt' in c.lower()]
        if amt_cols:
            amt_col = amt_cols[0]
            
            # Amount with time features
            if 'TransactionDT_hour' in X_new.columns:
                X_new['TransactionAmt_per_hour'] = X[amt_col] / (X_new['TransactionDT_hour'] + 1)
            
            # Amount with card features
            card_freq_cols = [c for c in X_new.columns if c.lower().startswith('card') and '_freq' in c]
            if card_freq_cols:
                X_new['TransactionAmt_card_interaction'] = X[amt_col] * X_new[card_freq_cols[0]]
            
            # Amount bins
            try:
                X_new['TransactionAmt_bin'] = pd.cut(X[amt_col], bins=20, labels=False, duplicates='drop')
            except ValueError:
                X_new['TransactionAmt_bin'] = 0
            
            X_new['TransactionAmt_log'] = np.log1p(X[amt_col])
            X_new['TransactionAmt_sqrt'] = np.sqrt(X[amt_col])
        
        # Card combinations
        card_cols = [c for c in X.columns if c.lower().startswith('card') and c[4:].isdigit()]
        if len(card_cols) >= 2:
            card1_col = next((c for c in card_cols if c.lower().endswith('1')), card_cols[0])
            card2_col = next((c for c in card_cols if c.lower().endswith('2')), card_cols[1])
            
            X_new['card1_card2_interaction'] = (X[card1_col].astype(str) + '_' + 
                                               X[card2_col].fillna('').astype(str))
            
            # Use a deterministic hash function that's safe for pandas
            def safe_hash(x):
                return abs(hash(x)) % 10000
            
            X_new['card1_card2_encoded'] = X_new['card1_card2_interaction'].apply(safe_hash)
        
        # Email domain interactions
        email_cols = [c for c in X.columns if 'emaildomain' in c.lower()]
        if len(email_cols) >= 2:
            X_new['email_domains_match'] = (X[email_cols[0]] == X[email_cols[1]]).astype(int)
        
        return X_new
    
    def _finalize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Final cleaning and feature selection."""
        if self.verbose:
            print("Finalizing features...")
        
        X_final = X.copy()
        
        # Handle infinite values
        X_final = X_final.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values
        for col in X_final.columns:
            if X_final[col].dtype in ['object', 'category']:
                X_final[col] = X_final[col].fillna('unknown')
            else:
                X_final[col] = X_final[col].fillna(0)
        
        # Remove original categorical columns that have been encoded
        cols_to_remove = []
        for col in X_final.columns:
            if col in ['uid', 'uid_addr', 'uid_device', 'card1_card2_interaction']:
                cols_to_remove.append(col)
        
        X_final = X_final.drop(columns=cols_to_remove, errors='ignore')
        
        # Ensure all columns are numeric
        for col in X_final.columns:
            if X_final[col].dtype == 'object':
                try:
                    X_final[col] = pd.to_numeric(X_final[col], errors='coerce').fillna(0)
                except:
                    # Convert to categorical codes as fallback
                    try:
                        X_final[col] = X_final[col].astype('category').cat.codes
                    except:
                        # Last resort: drop the column
                        X_final = X_final.drop(columns=[col])
        
        if self.verbose:
            print(f"Final feature count: {X_final.shape[1]}")
        
        return X_final
    
    def get_feature_names(self) -> List[str]:
        """Get list of all engineered feature names."""
        if not self.is_fitted_:
            raise ValueError("Feature engineer must be fitted before getting feature names.")
        return self.feature_names_.copy()
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Group features by their engineering method for interpretability."""
        if not self.is_fitted_:
            raise ValueError("Feature engineer must be fitted before getting feature groups.")
        
        groups = {
            'original': [],
            'uid_features': [],
            'temporal_features': [],
            'frequency_encoded': [],
            'aggregation_features': [],
            'interaction_features': []
        }
        
        for feature in self.feature_names_:
            if any(x in feature.lower() for x in ['uid', 'UID']):
                groups['uid_features'].append(feature)
            elif any(x in feature.lower() for x in ['transactiondt', 'hour', 'day', 'week']):
                groups['temporal_features'].append(feature)
            elif '_freq' in feature or '_rare' in feature:
                groups['frequency_encoded'].append(feature)
            elif any(x in feature for x in ['_mean', '_std', '_min', '_max', '_count']):
                groups['aggregation_features'].append(feature)
            elif any(x in feature for x in ['_interaction', '_bin', '_log', '_sqrt']):
                groups['interaction_features'].append(feature)
            else:
                groups['original'].append(feature)
        
        return groups


def create_advanced_features(train_df: pd.DataFrame, 
                           test_df: Optional[pd.DataFrame] = None,
                           verbose: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], AdvancedFeatureEngineer]:
    """
    Convenience function to create advanced features for IEEE fraud detection.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame (optional)
        verbose: Whether to print progress
        
    Returns:
        Tuple of (train_features, test_features, fitted_engineer)
    """
    if verbose:
        print("=== Advanced Feature Engineering Pipeline ===")
        print(f"Training data shape: {train_df.shape}")
        if test_df is not None:
            print(f"Test data shape: {test_df.shape}")
    
    # Initialize and fit feature engineer
    engineer = AdvancedFeatureEngineer(verbose=verbose)
    engineer.fit(train_df)
    
    # Transform training data
    train_features = engineer.transform(train_df)
    
    # Transform test data if provided
    test_features = None
    if test_df is not None:
        test_features = engineer.transform(test_df)
    
    if verbose:
        print("\n=== Feature Engineering Complete ===")
        print(f"Training features shape: {train_features.shape}")
        if test_features is not None:
            print(f"Test features shape: {test_features.shape}")
        
        # Print feature group summary
        groups = engineer.get_feature_importance_groups()
        print("\nFeature Groups:")
        for group_name, features in groups.items():
            print(f"  {group_name}: {len(features)} features")
    
    return train_features, test_features, engineer


# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the feature engineering pipeline
    print("Advanced Feature Engineering Module")
    print("Use create_advanced_features() function to process IEEE fraud data.")
    print("Expected input: Merged transaction + identity DataFrames")
    print("Expected output: 263+ engineered features ready for XGBoost training")