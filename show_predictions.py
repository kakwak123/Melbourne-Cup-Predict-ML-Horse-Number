import pandas as pd

pred_df = pd.read_csv('results.csv')
input_df = pd.read_csv('data/processed/2025_lineup.csv')

merged = pred_df.merge(input_df[['horse_number', 'horse_name', 'odds', 'weight', 'barrier']], 
                       on=['horse_number', 'horse_name'], how='left')

print('='*80)
print('2025 MELBOURNE CUP PREDICTIONS (with randomness enabled)')
print('='*80)
print('\nTop 10 Predictions:\n')
top10 = merged.head(10)
for idx, row in top10.iterrows():
    prob = row['top3_probability']
    odds_val = row['odds'] if pd.notna(row['odds']) else 0
    print(f"{int(row['horse_number']):2d}. {row['horse_name']:20s} | Odds: ${odds_val:6.2f} | Weight: {row['weight']:5.1f}kg | Barrier: {int(row['barrier']):2d} | Top-3 Prob: {prob:.4f}")

print('\n' + '-'*80)
print('\nProbability Statistics:')
print(f'  Range: {merged["top3_probability"].min():.4f} - {merged["top3_probability"].max():.4f}')
print(f'  Unique values: {merged["top3_probability"].nunique()}')
print(f'  Mean: {merged["top3_probability"].mean():.4f}')
print(f'  Std: {merged["top3_probability"].std():.4f}')

print('\n🎲 Top 3 Horses (with randomness):')
top3 = merged.head(3)
for idx, row in top3.iterrows():
    print(f"  {int(row['horse_number'])}. {row['horse_name']} (Prob: {row['top3_probability']:.4f})")

print('\n💡 Note: Run again to see different predictions due to randomness!')
