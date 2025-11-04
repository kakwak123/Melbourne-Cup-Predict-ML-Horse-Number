"""
Prediction module for Melbourne Cup top-3 finishers.

This module handles:
- Loading trained models
- Preprocessing input data
- Running ensemble predictions
- Converting finishing positions to top-3 probabilities
- Outputting predictions in required format
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
import warnings

from data_fetch import prepare_prediction_data, DataFetcher, DataPreprocessor
from train_models import XGBoostModel, LSTMModel

warnings.filterwarnings('ignore')


class PredictionEngine:
    """Main prediction engine combining XGBoost and LSTM models."""
    
    def __init__(self, model_dir: str = "models", xgb_weight: float = 0.6):
        """
        Initialize prediction engine.
        
        Args:
            model_dir: Directory containing saved models
            xgb_weight: Weight for XGBoost in ensemble (LSTM weight = 1 - xgb_weight)
        """
        self.model_dir = model_dir
        self.xgb_weight = xgb_weight
        self.lstm_weight = 1.0 - xgb_weight
        
        self.xgb_model = None
        self.lstm_model = None
        self.preprocessor = None
        
        self._load_models()
    
    def _load_models(self):
        """Load trained models and preprocessors."""
        try:
            # Load preprocessors
            self.preprocessor = DataPreprocessor()
            self.preprocessor.load_preprocessors()
            
            # Load XGBoost model
            xgb_path = os.path.join(self.model_dir, "xgboost_model.pkl")
            self.xgb_model = XGBoostModel(xgb_path)
            self.xgb_model.load()
            
            # Load LSTM model
            lstm_path = os.path.join(self.model_dir, "lstm_model.keras")
            self.lstm_model = LSTMModel(lstm_path)
            self.lstm_model.load()
            
            print("Models loaded successfully.")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Models not found. Please train models first using train_models.py. Error: {e}"
            )
    
    def _convert_to_top3_probabilities(self, positions: np.ndarray) -> np.ndarray:
        """
        Convert finishing positions to top-3 probabilities.
        
        Uses a softmax-like transformation where lower positions (better finishes)
        have higher probabilities.
        
        Args:
            positions: Predicted finishing positions
            
        Returns:
            Array of top-3 probabilities
        """
        # Convert positions to scores (lower position = higher score)
        # Add small epsilon to avoid division by zero
        scores = 1.0 / (positions + 0.1)
        
        # Apply softmax to get probabilities
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Scale to top-3 probability (sum of top 3 probabilities)
        # Get top 3 indices
        top3_indices = np.argsort(positions)[:3]
        
        # Create binary indicator for top 3
        top3_indicator = np.zeros_like(probabilities)
        top3_indicator[top3_indices] = 1
        
        # Calculate probability as sum of top 3 individual probabilities
        top3_probs = probabilities * top3_indicator
        top3_probs = top3_probs / np.sum(top3_probs) * np.sum(probabilities[top3_indices])
        
        return top3_probs
    
    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on input data.
        
        Args:
            input_data: DataFrame with horse data (can be raw or preprocessed)
            
        Returns:
            DataFrame with predictions including horse number, name, and top-3 probability
        """
        # Preprocess data if not already preprocessed
        if 'trainer_encoded' not in input_data.columns:
            processed_data = self.preprocessor.preprocess(input_data, fit=False)
        else:
            processed_data = input_data.copy()
        
        # Get feature columns
        feature_cols = self.preprocessor.feature_columns
        
        # XGBoost prediction
        X_xgb = processed_data[feature_cols]
        xgb_predictions = self.xgb_model.predict(X_xgb)
        
        # LSTM prediction (requires sequences)
        # For single race prediction, we'll use average features as sequence
        # This is a simplified approach - ideally we'd have historical sequences per horse
        try:
            # Create sequences from current data
            # Repeat each horse's features to create a sequence
            sequence_length = self.lstm_model.sequence_length
            X_lstm = []
            
            for idx in range(len(processed_data)):
                horse_features = processed_data.iloc[idx][feature_cols].values
                # Create sequence by repeating features
                sequence = np.tile(horse_features, (sequence_length, 1))
                X_lstm.append(sequence)
            
            X_lstm = np.array(X_lstm)
            lstm_predictions = self.lstm_model.predict(X_lstm)
        except Exception as e:
            print(f"Warning: LSTM prediction failed ({e}). Using XGBoost only.")
            lstm_predictions = xgb_predictions.copy()
            # Adjust weights to use only XGBoost
            self.xgb_weight = 1.0
            self.lstm_weight = 0.0
        
        # Ensemble prediction (weighted average)
        ensemble_predictions = (
            self.xgb_weight * xgb_predictions +
            self.lstm_weight * lstm_predictions
        )
        
        # Convert to top-3 probabilities
        top3_probs = self._convert_to_top3_probabilities(ensemble_predictions)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'horse_number': input_data['horse_number'].values,
            'horse_name': input_data['horse_name'].values,
            'predicted_position': ensemble_predictions,
            'top3_probability': top3_probs,
            'xgb_prediction': xgb_predictions,
            'lstm_prediction': lstm_predictions
        })
        
        # Sort by top-3 probability descending
        results = results.sort_values('top3_probability', ascending=False).reset_index(drop=True)
        
        return results
    
    def predict_top3(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict top 3 finishers with probabilities.
        
        Args:
            input_data: DataFrame with horse data
            
        Returns:
            DataFrame with top 3 predictions
        """
        predictions = self.predict(input_data)
        return predictions.head(3)[['horse_number', 'horse_name', 'top3_probability']]


def load_input_data(input_path: Optional[str] = None, year: Optional[int] = None) -> pd.DataFrame:
    """
    Load input data from file or generate mock data.
    
    Args:
        input_path: Path to input CSV/JSON file
        year: Year for mock data generation
        
    Returns:
        DataFrame with input data
    """
    if input_path and os.path.exists(input_path):
        if input_path.endswith('.json'):
            return pd.read_json(input_path)
        else:
            return pd.read_csv(input_path)
    else:
        # Generate mock data if no input file provided
        print(f"No input file provided. Generating mock data for year {year or 2024}...")
        fetcher = DataFetcher()
        return fetcher.generate_mock_data(num_horses=24, year=year or 2024)


def predict_melbourne_cup(input_path: Optional[str] = None, 
                         year: Optional[int] = None,
                         model_dir: str = "models",
                         output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Main prediction function for Melbourne Cup.
    
    Args:
        input_path: Path to input data file (optional)
        year: Year of Melbourne Cup (optional)
        model_dir: Directory containing trained models
        output_path: Path to save results (optional)
        
    Returns:
        DataFrame with predictions
    """
    # Load input data
    input_data = load_input_data(input_path, year)
    
    # Initialize prediction engine
    engine = PredictionEngine(model_dir=model_dir)
    
    # Make predictions
    predictions = engine.predict(input_data)
    
    # Save results if output path specified
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        predictions.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")
    
    return predictions


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict Melbourne Cup top-3 finishers')
    parser.add_argument('--input', type=str, help='Path to input data file')
    parser.add_argument('--year', type=int, help='Year of Melbourne Cup')
    parser.add_argument('--output', type=str, default='results.csv', help='Path to save results')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory containing models')
    
    args = parser.parse_args()
    
    predictions = predict_melbourne_cup(
        input_path=args.input,
        year=args.year,
        model_dir=args.model_dir,
        output_path=args.output
    )
    
    print("\nPredictions:")
    print(predictions[['horse_number', 'horse_name', 'top3_probability']].head(10))

