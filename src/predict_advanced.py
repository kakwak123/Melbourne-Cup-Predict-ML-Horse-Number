"""
Prediction script using advanced ML models
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetch import DataPreprocessor
from src.advanced_models import AdvancedModelTrainer

def predict_with_advanced_model(input_path: str, model_dir: str = "models") -> pd.DataFrame:
    """Make predictions using the best advanced model."""
    
    # Load best model
    model_name_path = os.path.join(model_dir, "best_model_name.pkl")
    if not os.path.exists(model_name_path):
        raise FileNotFoundError("Advanced models not trained. Run src/advanced_models.py first.")
    
    best_model_name = joblib.load(model_name_path)
    model_path = os.path.join(model_dir, "best_model.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading best model: {best_model_name}")
    model = joblib.load(model_path)
    scaler = joblib.load(os.path.join(model_dir, "advanced_scaler.pkl"))
    
    # Load and preprocess input data
    input_df = pd.read_csv(input_path)
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessors()
    processed_df = preprocessor.preprocess(input_df, fit=False)
    
    # Get feature columns
    feature_cols = preprocessor.feature_columns
    X = processed_df[feature_cols]
    
    # Scale features
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Make predictions
    if hasattr(model, 'predict_proba'):
        top3_probs = model.predict_proba(X_scaled)[:, 1]
    else:
        # For models without predict_proba, use decision function
        decision_scores = model.decision_function(X_scaled)
        # Normalize to probabilities using sigmoid
        top3_probs = 1 / (1 + np.exp(-decision_scores))
    
    # Create results
    results = pd.DataFrame({
        'horse_number': input_df['horse_number'].values,
        'horse_name': input_df['horse_name'].values,
        'top3_probability': top3_probs
    })
    
    results = results.sort_values('top3_probability', ascending=False).reset_index(drop=True)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict Melbourne Cup using advanced ML models")
    parser.add_argument('--input', type=str, default='data/processed/2025_lineup.csv',
                       help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='results_advanced.csv',
                       help='Path to save results CSV')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory containing trained models')
    
    args = parser.parse_args()
    
    try:
        predictions = predict_with_advanced_model(args.input, args.model_dir)
        
        print("\n" + "="*70)
        print("ADVANCED ML MODEL PREDICTIONS")
        print("="*70)
        print(f"\nTop 10 Predictions:")
        for idx, row in predictions.head(10).iterrows():
            print(f"  {idx+1:2d}. Horse {int(row['horse_number']):2d}: {row['horse_name']:20s} | Prob: {row['top3_probability']:.4f}")
        
        predictions.to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

