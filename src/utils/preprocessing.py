import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class FraudDataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and duplicates"""
        logger.info(f"Cleaning data with shape: {data.shape}")
        
        # Create a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Handle missing values - simple approach
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        categorical_columns = cleaned_data.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            if cleaned_data[col].isnull().sum() > 0:
                cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
        
        # Fill categorical missing values with 'unknown'
        for col in categorical_columns:
            if cleaned_data[col].isnull().sum() > 0:
                cleaned_data[col].fillna('unknown', inplace=True)
        
        # Remove duplicates
        cleaned_data = cleaned_data.drop_duplicates()
        
        logger.info(f"Data cleaned. Final shape: {cleaned_data.shape}")
        return cleaned_data

    def encode_categorical_features(self, data: pd.DataFrame, categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Encode categorical features using simple label encoding"""
        if categorical_columns is None:
            categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        encoded_data = data.copy()
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                encoded_data[col] = self.encoders[col].fit_transform(encoded_data[col].astype(str))
            else:
                # Simple handling of unseen categories
                try:
                    encoded_data[col] = self.encoders[col].transform(encoded_data[col].astype(str))
                except ValueError:
                    # Assign default value for unseen labels
                    encoded_data[col] = 0
        
        return encoded_data

    def scale_features(self, data: pd.DataFrame, method: str = 'standard', columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Scale numerical features"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        scaled_data = data.copy()
        
        for col in columns:
            if col not in self.scalers:
                if method == 'standard':
                    self.scalers[col] = StandardScaler()
                elif method == 'minmax':
                    self.scalers[col] = MinMaxScaler()
                else:
                    raise ValueError(f"Unknown scaling method: {method}")
                
                scaled_data[col] = self.scalers[col].fit_transform(scaled_data[[col]])
            else:
                scaled_data[col] = self.scalers[col].transform(scaled_data[[col]])
        
        return scaled_data

    def create_time_features(self, data: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
        """Create time-based features from datetime column"""
        enhanced_data = data.copy()
        
        if datetime_column in enhanced_data.columns:
            enhanced_data[datetime_column] = pd.to_datetime(enhanced_data[datetime_column])
            
            # Extract time features
            enhanced_data['hour'] = enhanced_data[datetime_column].dt.hour
            enhanced_data['day_of_week'] = enhanced_data[datetime_column].dt.dayofweek
            enhanced_data['month'] = enhanced_data[datetime_column].dt.month
            enhanced_data['is_weekend'] = enhanced_data['day_of_week'].isin([5, 6]).astype(int)
            
            # Create time-based risk indicators
            enhanced_data['is_night'] = enhanced_data['hour'].between(0, 6).astype(int)
            enhanced_data['is_business_hours'] = enhanced_data['hour'].between(9, 17).astype(int)
        
        return enhanced_data

    def transform_data(self, data: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """Complete data transformation pipeline"""
        logger.info("Starting data transformation pipeline")
        
        # Step 1: Clean data
        transformed_data = self.clean_data(data)
        
        # Step 2: Create time features if datetime column exists
        datetime_columns = transformed_data.select_dtypes(include=['datetime64']).columns
        if len(datetime_columns) > 0:
            for dt_col in datetime_columns:
                transformed_data = self.create_time_features(transformed_data, dt_col)
        
        # Step 3: Encode categorical features (except target)
        categorical_columns = transformed_data.select_dtypes(include=['object']).columns.tolist()
        if target_column and target_column in categorical_columns:
            categorical_columns.remove(target_column)
        
        if categorical_columns:
            transformed_data = self.encode_categorical_features(transformed_data, categorical_columns)
        
        # Step 4: Scale features 
        numeric_columns = transformed_data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numeric_columns:
            numeric_columns.remove(target_column)
        
        if numeric_columns:
            transformed_data = self.scale_features(transformed_data, columns=numeric_columns)
        
        self.feature_names = transformed_data.columns.tolist()
        logger.info(f"Data transformation completed. Final shape: {transformed_data.shape}")
        
        return transformed_data

    def prepare_for_training(self, data: pd.DataFrame, target_column: str, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        # Transform data
        transformed_data = self.transform_data(data, target_column)
        
        # Separate features and target
        X = transformed_data.drop(columns=[target_column])
        y = transformed_data[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        logger.info(f"Feature count: {X_train.shape[1]}")
        
        return X_train.values, X_test.values, y_train.values, y_test.values

    def prepare_for_prediction(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare new data for prediction"""
        # Apply same transformations as training data
        transformed_data = data.copy()
        
        # Clean data
        transformed_data = self.clean_data(transformed_data)
        
        # Create time features
        datetime_columns = transformed_data.select_dtypes(include=['datetime64']).columns
        for dt_col in datetime_columns:
            transformed_data = self.create_time_features(transformed_data, dt_col)
        
        # Encode categorical features
        categorical_columns = transformed_data.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            transformed_data = self.encode_categorical_features(transformed_data, categorical_columns)
        
        # Scale numerical features
        numeric_columns = transformed_data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            transformed_data = self.scale_features(transformed_data, columns=numeric_columns)
        
        # Ensure same feature order as training
        if self.feature_names:
            # Add missing columns with zeros
            for col in self.feature_names:
                if col not in transformed_data.columns:
                    transformed_data[col] = 0
            
            # Reorder columns to match training
            transformed_data = transformed_data[self.feature_names]
        
        return transformed_data.values


# Legacy functions for backward compatibility
def clean_data(data):
    """Legacy function - use FraudDataPreprocessor instead"""
    preprocessor = FraudDataPreprocessor()
    return preprocessor.clean_data(data)

def transform_data(data):
    """Legacy function - use FraudDataPreprocessor instead"""
    if isinstance(data, pd.DataFrame):
        preprocessor = FraudDataPreprocessor()
        return preprocessor.transform_data(data)
    else:
        # Simple standardization for numpy arrays
        return (data - data.mean()) / data.std()