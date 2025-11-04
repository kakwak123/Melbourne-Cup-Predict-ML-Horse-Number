"""
Model training and evaluation module for Melbourne Cup predictions.

This module implements:
- XGBoost regression model for finishing position prediction
- LSTM neural network for temporal pattern recognition
- Model evaluation metrics (MAE, RMSE, top-3 accuracy)
- Model persistence and loading
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import warnings
from typing import Tuple, Dict, Optional
import json

from data_fetch import prepare_training_data, DataPreprocessor

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class XGBoostModel:
    """XGBoost regression model for finishing position prediction."""
    
    def __init__(self, model_path: str = "models/xgboost_model.pkl"):
        """
        Initialize XGBoost model.
        
        Args:
            model_path: Path to save/load the model
        """
        self.model_path = model_path
        self.model = None
        self.best_params = None
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Dict:
        """
        Train XGBoost model with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary with training metrics
        """
        print("Training XGBoost model...")
        
        # Hyperparameter grid for tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Base model
        base_model = xgb.XGBRegressor(
            random_state=42,
            objective='reg:squarederror',
            eval_metric='rmse'
        )
        
        # Perform grid search
        print("Performing hyperparameter tuning...")
        if X_val is not None and y_val is not None:
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
        else:
            # Use default parameters if no validation set
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.9,
                random_state=42,
                objective='reg:squarederror'
            )
            self.model.fit(X_train, y_train)
            self.best_params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.9
            }
        
        print(f"Best parameters: {self.best_params}")
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        
        metrics = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'best_params': self.best_params
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            metrics['val_mae'] = val_mae
            metrics['val_rmse'] = val_rmse
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict finishing positions.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Predicted finishing positions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def save(self):
        """Save model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"XGBoost model saved to {self.model_path}")
    
    def load(self):
        """Load model from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"XGBoost model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")


class LSTMModel:
    """LSTM neural network for temporal pattern recognition."""
    
    def __init__(self, model_path: str = "models/lstm_model.keras", 
                 sequence_length: int = 5):
        """
        Initialize LSTM model.
        
        Args:
            model_path: Path to save/load the model
            sequence_length: Length of input sequences
        """
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.model = None
        self.feature_dim = None
    
    def prepare_sequences(self, df: pd.DataFrame, target_col: str = 'finishing_position') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM input.
        
        Args:
            df: DataFrame with historical data
            target_col: Name of target column
            
        Returns:
            Tuple of (X_sequences, y_targets)
        """
        # Group by horse to create sequences
        grouped = df.groupby('horse_name')
        
        sequences = []
        targets = []
        
        for horse_name, group in grouped:
            # Sort by year/race date if available
            if 'year' in group.columns:
                group = group.sort_values('year')
            
            # Extract features
            feature_cols = [col for col in group.columns 
                           if col not in ['horse_name', 'horse_number', 'year', 
                                         'trainer', 'jockey', 'track_condition',
                                         'recent_race_times', target_col]]
            
            # Get numerical features
            features = group[feature_cols].values
            
            if len(features) >= self.sequence_length:
                # Create sequences
                for i in range(len(features) - self.sequence_length + 1):
                    seq = features[i:i + self.sequence_length]
                    sequences.append(seq)
                    # Use the target from the last element of the sequence
                    if target_col in group.columns:
                        targets.append(group.iloc[i + self.sequence_length - 1][target_col])
                    else:
                        targets.append(None)
            else:
                # Pad shorter sequences
                padding = np.zeros((self.sequence_length - len(features), len(feature_cols)))
                padded_features = np.vstack([padding, features])
                sequences.append(padded_features)
                if target_col in group.columns:
                    targets.append(group.iloc[-1][target_col])
                else:
                    targets.append(None)
        
        X = np.array(sequences)
        y = np.array(targets)
        
        # Filter out None targets
        valid_indices = ~pd.isna(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        self.feature_dim = X.shape[2]
        
        return X, y.astype(float)
    
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input sequences (sequence_length, feature_dim)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Masking(mask_value=0.0, input_shape=input_shape),
            LSTM(64, return_sequences=True, dropout=0.2),
            LSTM(32, return_sequences=False, dropout=0.2),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='linear')  # Regression output
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        Train LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Dictionary with training history
        """
        print("Training LSTM model...")
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        print(f"Model architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss',
                         patience=10, restore_best_weights=True),
            ModelCheckpoint(self.model_path, monitor='val_loss' if X_val is not None else 'loss',
                          save_best_only=True, verbose=1)
        ]
        
        # Train model
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        train_pred = self.model.predict(X_train, verbose=0)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        
        metrics = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'history': history.history
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val, verbose=0)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            metrics['val_mae'] = val_mae
            metrics['val_rmse'] = val_rmse
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict finishing positions.
        
        Args:
            X: Input sequences
            
        Returns:
            Predicted finishing positions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X, verbose=0).flatten()
    
    def save(self):
        """Save model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        if self.model is not None:
            self.model.save(self.model_path)
            print(f"LSTM model saved to {self.model_path}")
    
    def load(self):
        """Load model from disk."""
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print(f"LSTM model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")


def calculate_top3_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate top-3 ranking accuracy.
    
    Args:
        y_true: True finishing positions
        y_pred: Predicted finishing positions
        
    Returns:
        Accuracy score (percentage of correctly predicted top-3)
    """
    # Get top 3 from true and predicted
    true_top3 = set(np.argsort(y_true)[:3])
    pred_top3 = set(np.argsort(y_pred)[:3])
    
    # Calculate overlap
    overlap = len(true_top3.intersection(pred_top3))
    return overlap / 3.0


def evaluate_models(xgb_model: XGBoostModel, lstm_model: LSTMModel,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   X_test_lstm: Optional[np.ndarray] = None) -> Dict:
    """
    Evaluate both models on test set.
    
    Args:
        xgb_model: Trained XGBoost model
        lstm_model: Trained LSTM model
        X_test: Test features (for XGBoost)
        y_test: Test targets
        X_test_lstm: Test sequences (for LSTM, optional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\nEvaluating models...")
    
    results = {}
    
    # Evaluate XGBoost
    xgb_pred = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_top3 = calculate_top3_accuracy(y_test.values, xgb_pred)
    
    results['xgboost'] = {
        'mae': xgb_mae,
        'rmse': xgb_rmse,
        'top3_accuracy': xgb_top3
    }
    
    print(f"\nXGBoost Results:")
    print(f"  MAE: {xgb_mae:.4f}")
    print(f"  RMSE: {xgb_rmse:.4f}")
    print(f"  Top-3 Accuracy: {xgb_top3:.4f}")
    
    # Evaluate LSTM if sequences available
    if X_test_lstm is not None:
        lstm_pred = lstm_model.predict(X_test_lstm)
        lstm_mae = mean_absolute_error(y_test, lstm_pred)
        lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))
        lstm_top3 = calculate_top3_accuracy(y_test.values, lstm_pred)
        
        results['lstm'] = {
            'mae': lstm_mae,
            'rmse': lstm_rmse,
            'top3_accuracy': lstm_top3
        }
        
        print(f"\nLSTM Results:")
        print(f"  MAE: {lstm_mae:.4f}")
        print(f"  RMSE: {lstm_rmse:.4f}")
        print(f"  Top-3 Accuracy: {lstm_top3:.4f}")
    
    return results


def train_all_models(data_dir: str = "data", test_size: float = 0.2, 
                     save_models: bool = True) -> Tuple[XGBoostModel, LSTMModel, Dict]:
    """
    Train both XGBoost and LSTM models.
    
    Args:
        data_dir: Base directory for data
        test_size: Proportion of data to use for testing
        save_models: Whether to save trained models
        
    Returns:
        Tuple of (XGBoost model, LSTM model, evaluation results)
    """
    print("Preparing training data...")
    X, y = prepare_training_data(data_dir=data_dir)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Further split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train XGBoost model
    xgb_model = XGBoostModel()
    xgb_metrics = xgb_model.train(X_train_split, y_train_split, X_val, y_val)
    print(f"\nXGBoost Training Metrics: {xgb_metrics}")
    
    # Prepare LSTM sequences
    # We need to recreate sequences from original data
    from data_fetch import DataFetcher, DataPreprocessor
    
    # Load preprocessors (they were saved in prepare_training_data)
    try:
        preprocessor = DataPreprocessor()
        preprocessor.load_preprocessors()
    except FileNotFoundError:
        # If preprocessors don't exist, create them again
        print("Preprocessors not found, recreating...")
        fetcher = DataFetcher(data_dir)
        historical_df = fetcher.load_historical_data()
        training_df = historical_df[historical_df['finishing_position'].notna()].copy()
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.preprocess(training_df, fit=True)
        preprocessor.save_preprocessors()
    else:
        fetcher = DataFetcher(data_dir)
        historical_df = fetcher.load_historical_data()
        training_df = historical_df[historical_df['finishing_position'].notna()].copy()
        processed_df = preprocessor.preprocess(training_df, fit=False)
    
    # Split for LSTM
    train_indices, test_indices = train_test_split(
        processed_df.index, test_size=test_size, random_state=42
    )
    train_df_lstm = processed_df.loc[train_indices]
    test_df_lstm = processed_df.loc[test_indices]
    
    # Prepare sequences
    lstm_model = LSTMModel()
    X_train_lstm, y_train_lstm = lstm_model.prepare_sequences(train_df_lstm)
    X_test_lstm, y_test_lstm = lstm_model.prepare_sequences(test_df_lstm)
    
    # Split training sequences for validation
    val_size = int(len(X_train_lstm) * 0.2)
    X_train_lstm_split = X_train_lstm[:-val_size]
    X_val_lstm = X_train_lstm[-val_size:]
    y_train_lstm_split = y_train_lstm[:-val_size]
    y_val_lstm = y_train_lstm[-val_size:]
    
    # Train LSTM model
    lstm_metrics = lstm_model.train(
        X_train_lstm_split, y_train_lstm_split,
        X_val_lstm, y_val_lstm,
        epochs=50, batch_size=32
    )
    print(f"\nLSTM Training Metrics: {lstm_metrics}")
    
    # Map test indices back for evaluation
    y_test_aligned = y_test  # Using the same test set
    
    # Evaluate both models
    results = evaluate_models(
        xgb_model, lstm_model,
        X_test, y_test_aligned,
        X_test_lstm
    )
    
    # Save models
    if save_models:
        xgb_model.save()
        lstm_model.save()
        
        # Save evaluation results
        results_path = os.path.join(data_dir, "..", "models", "evaluation_results.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nEvaluation results saved to {results_path}")
    
    return xgb_model, lstm_model, results


if __name__ == "__main__":
    print("Training Melbourne Cup prediction models...")
    xgb_model, lstm_model, results = train_all_models()
    print("\nTraining complete!")
    print("\nFinal Evaluation Results:")
    print(json.dumps(results, indent=2))

