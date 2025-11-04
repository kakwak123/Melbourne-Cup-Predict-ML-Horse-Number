import pandas as pd
import joblib
import numpy as np
from src.data_fetch import DataPreprocessor

# Load model and preprocessor
model = joblib.load('models/logistic_model.pkl')
feature_cols = joblib.load('models/feature_columns.pkl')

# Load input data
input_df = pd.read_csv('data/processed/2025_lineup.csv')

# Preprocess
preprocessor = DataPreprocessor()
preprocessor.load_preprocessors()
processed_df = preprocessor.preprocess(input_df, fit=False)

# Check Horse 14 vs Horse 1
horse14 = processed_df[processed_df['horse_number'] == 14].iloc[0]
horse1 = processed_df[processed_df['horse_number'] == 1].iloc[0]

print('='*80)
print('FEATURE COMPARISON: Horse 14 (Half Yours - Actual Winner) vs Horse 1 (AL RIFFA - Our #1)')
print('='*80)

print('\nRaw Features:')
print('-'*80)
print(f'Horse 14: Odds=${input_df[input_df["horse_number"]==14]["odds"].iloc[0]:.2f}, Weight={input_df[input_df["horse_number"]==14]["weight"].iloc[0]:.1f}kg')
print(f'Horse 1:  Odds=${input_df[input_df["horse_number"]==1]["odds"].iloc[0]:.2f}, Weight={input_df[input_df["horse_number"]==1]["weight"].iloc[0]:.1f}kg')

print('\nProcessed Features (after scaling):')
print('-'*80)
for feat in feature_cols:
    val14 = horse14[feat] if feat in horse14.index else 'N/A'
    val1 = horse1[feat] if feat in horse1.index else 'N/A'
    print(f'{feat:20s}: Horse 14={val14:8.4f} | Horse 1={val1:8.4f}')

# Get model predictions
X14 = processed_df[processed_df['horse_number'] == 14][feature_cols]
X1 = processed_df[processed_df['horse_number'] == 1][feature_cols]

prob14 = model.predict_proba(X14)[0, 1]
prob1 = model.predict_proba(X1)[0, 1]

print('\nModel Predictions:')
print('-'*80)
print(f'Horse 14 probability: {prob14:.4f}')
print(f'Horse 1 probability: {prob1:.4f}')

# Check feature contributions
coefs = model.coef_[0]
print('\nFeature Contributions (coefficient * feature_value):')
print('-'*80)
for feat, coef in zip(feature_cols, coefs):
    contrib14 = coef * horse14[feat] if feat in horse14.index else 0
    contrib1 = coef * horse1[feat] if feat in horse1.index else 0
    print(f'{feat:20s}: Horse 14={contrib14:8.4f} | Horse 1={contrib1:8.4f}')

print('\n' + '='*80)
print('INTERCEPT:', model.intercept_[0])
print('='*80)

