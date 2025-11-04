"""
Simplified prediction module for Melbourne Cup top-3 finishers.

This module handles:
- Loading trained model
- Preprocessing input data
- Running predictions
- Outputting predictions in required format
"""

import os
import numpy as np
import pandas as pd
from typing import Optional
import warnings

from data_fetch import prepare_prediction_data, DataFetcher, DataPreprocessor
from train_models import SimpleModel

warnings.filterwarnings('ignore')


class PredictionEngine:
    """Simple prediction engine using Logistic Regression."""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize prediction engine.
        
        Args:
            model_dir: Directory containing saved models
        """
        self.model_dir = model_dir
        self.model = None
        self.preprocessor = None
        
        self._load_models()
    
    def _load_models(self):
        """Load trained model and preprocessors."""
        try:
            # Load preprocessors
            self.preprocessor = DataPreprocessor()
            self.preprocessor.load_preprocessors()
            
            # Load model
            model_path = os.path.join(self.model_dir, "logistic_model.pkl")
            self.model = SimpleModel(model_path)
            self.model.load()
            
            print("Model loaded successfully.")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model not found. Please train model first using train_models.py. Error: {e}"
            )
    
    def predict(self, input_data: pd.DataFrame, add_randomness: bool = True, randomness_factor: float = 0.01) -> pd.DataFrame:
        """
        Make predictions on input data.
        
        Args:
            input_data: DataFrame with horse data (can be raw or preprocessed)
            add_randomness: Whether to add random variation to predictions
            randomness_factor: Factor controlling randomness (0.0-1.0, default 0.01 = 1% subtle variation)
            
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
        if feature_cols is None:
            raise ValueError("Feature columns not set. Preprocessor may not have been fitted.")
        
        # Filter to only columns that exist
        available_cols = [col for col in feature_cols if col in processed_data.columns]
        if len(available_cols) != len(feature_cols):
            missing = set(feature_cols) - set(available_cols)
            print(f"Warning: Missing columns: {missing}")
        
        # Make predictions
        X = processed_data[available_cols]
        top3_probs = self.model.predict_proba_top3(X)
        
        # Add subtle randomness for fun if requested
        if add_randomness:
            np.random.seed(None)  # Use current time as seed for randomness
            # Add subtle random noise (much smaller - scale by std instead of mean)
            noise_std = randomness_factor * top3_probs.std()  # Scale by actual std
            noise = np.random.normal(0, noise_std, size=len(top3_probs))
            top3_probs = top3_probs + noise
            
            # Ensure probabilities stay in valid range [0, 1]
            top3_probs = np.clip(top3_probs, 0.01, 0.99)
            
            # Subtle renormalization - keep it close to original
            original_sum = top3_probs.sum()
            if original_sum > 0:
                # Only adjust slightly (max 5% change)
                target_sum = len(top3_probs) * top3_probs.mean()  # Maintain average
                adjustment_factor = min(1.05, max(0.95, target_sum / original_sum))
                top3_probs = top3_probs * adjustment_factor
        
        # Create results DataFrame
        results = pd.DataFrame({
            'horse_number': input_data['horse_number'].values,
            'horse_name': input_data['horse_name'].values,
            'top3_probability': top3_probs
        })
        
        # Sort by top-3 probability descending
        results = results.sort_values('top3_probability', ascending=False).reset_index(drop=True)
        
        return results
    
    def predict_top3(self, input_data: pd.DataFrame, add_randomness: bool = True) -> pd.DataFrame:
        """
        Predict top 3 finishers with probabilities.
        
        Args:
            input_data: DataFrame with horse data
            add_randomness: Whether to add random variation to predictions
            
        Returns:
            DataFrame with top 3 predictions
        """
        predictions = self.predict(input_data, add_randomness=add_randomness)
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
                         output_path: Optional[str] = None,
                         add_randomness: bool = True,
                         randomness_factor: float = 0.01) -> pd.DataFrame:
    """
    Main prediction function for Melbourne Cup.
    
    Args:
        input_path: Path to input data file (optional)
        year: Year of Melbourne Cup (optional)
        model_dir: Directory containing trained models
        output_path: Path to save results (optional)
        add_randomness: Whether to add random variation to predictions
        randomness_factor: Factor controlling randomness (0.0-1.0)
        
    Returns:
        DataFrame with predictions
    """
    # Load input data
    input_data = load_input_data(input_path, year)
    
    # Initialize prediction engine
    engine = PredictionEngine(model_dir=model_dir)
    
    # Make predictions
    predictions = engine.predict(input_data, add_randomness=add_randomness, randomness_factor=randomness_factor)
    
    # Save results if output path specified
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        # Ensure sorted by probability descending before saving
        predictions = predictions.sort_values('top3_probability', ascending=False).reset_index(drop=True)
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
