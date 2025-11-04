import pandas as pd
import glob
import os

# Load all historical data
data_dir = 'data/historical'
files = glob.glob(os.path.join(data_dir, 'melbourne_cup_*.csv'))

all_data = []
for f in files:
    df = pd.read_csv(f)
    all_data.append(df)

historical = pd.concat(all_data, ignore_index=True)

print('HISTORICAL DATA PATTERNS:')
print('='*70)
print(f'Total horses: {len(historical)}')

# Check top 3 finishers
top3 = historical[historical['finishing_position'] <= 3].copy()

print(f'\nTop 3 Finishers Analysis:')
print(f'  Count: {len(top3)}')
if 'odds' in top3.columns:
    print(f'  Average Odds: ${top3["odds"].mean():.2f}')
    print(f'  Median Odds: ${top3["odds"].median():.2f}')
print(f'  Average Weight: {top3["weight"].mean():.1f}kg')
print(f'  Average Barrier: {top3["barrier"].mean():.1f}')

# Check all finishers
print(f'\nAll Finishers Analysis:')
if 'odds' in historical.columns:
    print(f'  Average Odds: ${historical["odds"].mean():.2f}')
    print(f'  Median Odds: ${historical["odds"].median():.2f}')
print(f'  Average Weight: {historical["weight"].mean():.1f}kg')
print(f'  Average Barrier: {historical["barrier"].mean():.1f}')

# Check correlation
if 'odds' in historical.columns:
    corr = historical['odds'].corr(historical['finishing_position'])
    print(f'\nOdds vs Finishing Position Correlation: {corr:.4f}')
    print(f'  (Should be POSITIVE - higher odds = worse position)')

# Check if favorites (low odds) win more
if 'odds' in historical.columns:
    favorites = historical[historical['odds'] <= 10].copy()
    favorite_top3_rate = len(favorites[favorites['finishing_position'] <= 3]) / len(favorites)
    print(f'\nFavorites (odds <= $10) Top-3 Rate: {favorite_top3_rate:.2%}')
    
    longshots = historical[historical['odds'] > 30].copy()
    longshot_top3_rate = len(longshots[longshots['finishing_position'] <= 3]) / len(longshots) if len(longshots) > 0 else 0
    print(f'Longshots (odds > $30) Top-3 Rate: {longshot_top3_rate:.2%}')

