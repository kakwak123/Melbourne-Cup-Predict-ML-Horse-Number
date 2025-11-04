"""
Simplified model training module using Logistic Regression for Melbourne Cup predictions.

This module implements:
- Logistic Regression for top-3 classification
- Model evaluation metrics
- Model persistence and loading
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
from typing import Tuple, Dict, Optional
import json

from data_fetch import prepare_training_data, DataPreprocessor

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)


class SimpleModel:
    """Simple Logistic Regression model for top-3 prediction."""
    
    def __init__(self, model_path: str = "models/logistic_model.pkl"):
        """
        Initialize Logistic Regression model.
        
        Args:
            model_path: Path to save/load the model
        """
        self.model_path = model_path
        self.model = None
        self.best_params = None
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Dict:
        """
        Train Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets (finishing positions)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary with training metrics
        """
        print("Training Logistic Regression model...")
        
        # Convert to binary classification: top-3 (1) vs not top-3 (0)
        y_train_binary = (y_train <= 3).astype(int)
        
        # Hyperparameter grid - favor L2 and higher C to allow stronger feature weights
        # Higher C values allow the model to weight odds more heavily (favorites win more!)
        # Use 'roc_auc' scoring to better handle imbalanced data
        param_grid = {
            'C': [100.0, 200.0, 500.0, 1000.0],  # Very high C = minimal regularization = strong odds weighting
            'penalty': ['l2'],  # L2 keeps all features (ridge)
            'solver': ['lbfgs', 'liblinear'],
            'class_weight': ['balanced', None]  # Handle class imbalance
        }
        
        # Base model
        base_model = LogisticRegression(random_state=42, max_iter=1000)
        
        print("Performing hyperparameter tuning...")
        if X_val is not None and y_val is not None:
            y_val_binary = (y_val <= 3).astype(int)
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=3,
                scoring='roc_auc',  # Better for imbalanced data than accuracy
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train_binary)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
        else:
            # Use default parameters with L2 regularization - higher C for stronger odds weighting
            # Use class_weight='balanced' to handle imbalanced data
            self.model = LogisticRegression(
                C=200.0,  # Higher C to allow stronger feature weights (especially for odds)
                penalty='l2',  # L2 regularization keeps all features
                solver='lbfgs',
                class_weight='balanced',  # Handle class imbalance (only ~13% are top-3)
                random_state=42,
                max_iter=2000  # More iterations for convergence
            )
            self.model.fit(X_train, y_train_binary)
            self.best_params = {
                'C': 200.0,
                'penalty': 'l2',
                'solver': 'lbfgs',
                'class_weight': 'balanced'
            }
        
        print(f"Best parameters: {self.best_params}")
        
        # Evaluate on training set
        train_pred_binary = self.model.predict(X_train)
        train_acc = accuracy_score(y_train_binary, train_pred_binary)
        
        # Get probabilities for top-3
        train_probs = self.model.predict_proba(X_train)[:, 1]  # Probability of being top-3
        
        metrics = {
            'train_accuracy': train_acc,
            'train_top3_prob_mean': train_probs.mean(),
            'best_params': self.best_params
        }
        
        if X_val is not None and y_val is not None:
            y_val_binary = (y_val <= 3).astype(int)
            val_pred_binary = self.model.predict(X_val)
            val_acc = accuracy_score(y_val_binary, val_pred_binary)
            metrics['val_accuracy'] = val_acc
        
        return metrics
    
    def predict_proba_top3(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of finishing in top-3.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of top-3 probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)[:, 1]  # Probability of class 1 (top-3)
    
    def save(self):
        """Save model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load(self):
        """Load model from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")


def calculate_top3_accuracy(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calculate top-3 ranking accuracy.
    
    Args:
        y_true: True finishing positions
        y_pred_proba: Predicted probabilities of top-3
        
    Returns:
        Accuracy score (percentage of correctly predicted top-3)
    """
    # Get top 3 from true and predicted
    true_top3 = set(np.argsort(y_true)[:3])
    pred_top3 = set(np.argsort(y_pred_proba)[-3:])  # Top 3 highest probabilities
    
    # Calculate overlap
    overlap = len(true_top3.intersection(pred_top3))
    return overlap / 3.0


def evaluate_model(model: SimpleModel, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\nEvaluating model...")
    
    # Get predictions
    y_test_binary = (y_test <= 3).astype(int)
    y_pred_binary = model.model.predict(X_test)
    y_pred_proba = model.predict_proba_top3(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    top3_acc = calculate_top3_accuracy(y_test.values, y_pred_proba)
    
    # For regression metrics, convert probabilities back to positions
    # Sort by probability descending and assign ranks
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_pred_positions = np.zeros_like(y_test.values)
    for rank, idx in enumerate(sorted_indices, 1):
        y_pred_positions[idx] = rank
    
    mae = mean_absolute_error(y_test.values, y_pred_positions)
    rmse = np.sqrt(mean_squared_error(y_test.values, y_pred_positions))
    
    results = {
        'accuracy': accuracy,
        'top3_accuracy': top3_acc,
        'mae': mae,
        'rmse': rmse
    }
    
    print(f"\nModel Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Top-3 Accuracy: {top3_acc:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    return results


def train_model(data_dir: str = "data", test_size: float = 0.2, 
               save_model: bool = True) -> Tuple[SimpleModel, Dict]:
    """
    Train Logistic Regression model.
    
    Args:
        data_dir: Base directory for data
        test_size: Proportion of data to use for testing
        save_model: Whether to save trained model
        
    Returns:
        Tuple of (model, evaluation results)
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
    
    # Train model
    model = SimpleModel()
    train_metrics = model.train(X_train_split, y_train_split, X_val, y_val)
    print(f"\nTraining Metrics: {train_metrics}")
    
    # Evaluate on test set
    results = evaluate_model(model, X_test, y_test)
    
    # Save model
    if save_model:
        model.save()
        
        # Save evaluation results
        results_path = os.path.join(data_dir, "..", "models", "evaluation_results.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nEvaluation results saved to {results_path}")
    
    return model, results


if __name__ == "__main__":
    print("Training Melbourne Cup prediction model...")
    model, results = train_model()
    print("\nTraining complete!")
    print("\nFinal Evaluation Results:")
    print(json.dumps(results, indent=2))
