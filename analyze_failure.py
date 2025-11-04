import pandas as pd
import numpy as np

# Actual results
actual = {14: 1, 20: 2, 7: 3, 21: 4}

# Load predictions and input
pred_df = pd.read_csv('results.csv')
input_df = pd.read_csv('data/processed/2025_lineup.csv')

merged = pred_df.merge(input_df[['horse_number', 'horse_name', 'odds', 'weight', 'barrier', 'age', 'trainer', 'jockey']], 
                       on=['horse_number', 'horse_name'], how='left')

print('='*80)
print('ACTUAL WINNERS vs OUR PREDICTIONS')
print('='*80)
print('\nActual Top 3:')
for horse_num in [14, 20, 7]:
    row = merged[merged['horse_number'] == horse_num].iloc[0]
    pred_rank = merged[merged['horse_number'] == horse_num].index[0] + 1
    print(f"\nHorse {horse_num}: {row['horse_name']}")
    print(f"  Actual Position: {actual[horse_num]}")
    print(f"  Our Prediction Rank: {pred_rank}")
    print(f"  Our Probability: {row['top3_probability']:.4f}")
    print(f"  Odds: ${row['odds']:.2f} | Weight: {row['weight']:.1f}kg | Barrier: {int(row['barrier'])} | Age: {int(row['age'])}")

print('\n' + '-'*80)
print('Our Top 3 Predictions:')
top3 = merged.head(3)
for idx, row in top3.iterrows():
    actual_pos = actual.get(int(row['horse_number']), 'N/A')
    print(f"\nHorse {int(row['horse_number'])}: {row['horse_name']}")
    print(f"  Our Rank: {idx + 1} | Actual Position: {actual_pos}")
    print(f"  Probability: {row['top3_probability']:.4f}")
    print(f"  Odds: ${row['odds']:.2f} | Weight: {row['weight']:.1f}kg | Barrier: {int(row['barrier'])} | Age: {int(row['age'])}")

print('\n' + '-'*80)
print('PATTERN ANALYSIS:')
print('-'*80)

# Analyze actual winners
winners = merged[merged['horse_number'].isin([14, 20, 7])]
print(f"\nActual Winners Stats:")
print(f"  Average Odds: ${winners['odds'].mean():.2f}")
print(f"  Average Weight: {winners['weight'].mean():.1f}kg")
print(f"  Average Barrier: {winners['barrier'].mean():.1f}")
print(f"  Average Age: {winners['age'].mean():.1f}")

# Compare to our predictions
our_top3 = merged.head(3)
print(f"\nOur Top 3 Predictions Stats:")
print(f"  Average Odds: ${our_top3['odds'].mean():.2f}")
print(f"  Average Weight: {our_top3['weight'].mean():.1f}kg")
print(f"  Average Barrier: {our_top3['barrier'].mean():.1f}")
print(f"  Average Age: {our_top3['age'].mean():.1f}")

print('\n' + '-'*80)
print('Key Issues:')
print('-'*80)
print('1. Horse 14 (Half Yours) was favorite ($6.5) but ranked 14th by us!')
print('2. Our top predictions have higher average odds than actual winners')
print('3. Weight might be more important - winners were lighter (51.5-54.5kg)')
print('4. Barrier might not be as important as we thought')

