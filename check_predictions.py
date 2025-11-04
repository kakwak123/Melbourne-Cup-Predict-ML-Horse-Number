import pandas as pd

pred_df = pd.read_csv('results.csv')
input_df = pd.read_csv('data/processed/2025_lineup.csv')

merged = pred_df.merge(input_df[['horse_number', 'horse_name', 'odds', 'weight', 'barrier']], 
                       on=['horse_number', 'horse_name'], how='left')

print('='*80)
print('PREDICTIONS vs ODDS (should be inversely related)')
print('='*80)
print('\nTop 10 by Probability:')
top10 = merged.nlargest(10, 'top3_probability')[['horse_number', 'horse_name', 'odds', 'weight', 'top3_probability']]
for idx, row in top10.iterrows():
    odds_val = row['odds'] if pd.notna(row['odds']) else 'N/A'
    print(f"{int(row['horse_number']):2d}. {row['horse_name']:20s} | Odds: ${odds_val:7s} | Weight: {row['weight']:5.1f}kg | Prob: {row['top3_probability']:.4f}")

print('\n' + '-'*80)
print('Lowest Odds (Favorites):')
lowest = merged.nsmallest(5, 'odds')[['horse_number', 'horse_name', 'odds', 'top3_probability']]
print(lowest.to_string(index=False))

print('\n' + '-'*80)
prob_min = merged['top3_probability'].min()
prob_max = merged['top3_probability'].max()
odds_min = merged['odds'].min()
odds_max = merged['odds'].max()
corr = merged['odds'].corr(merged['top3_probability'])

print(f'\nProbability Range: {prob_min:.4f} - {prob_max:.4f}')
print(f'Odds Range: {odds_min:.2f} - {odds_max:.2f}')
print(f'Correlation: {corr:.4f}')

