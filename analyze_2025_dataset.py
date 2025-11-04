import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/2025_lineup.csv')

# Parse recent race times
df['recent_race_times'] = df['recent_race_times'].apply(eval)

# Actual winners
winners = [14, 20, 7, 21]
df['is_winner'] = df['horse_number'].isin(winners)

print('='*80)
print('COMPREHENSIVE DATASET ANALYSIS - 2025 Melbourne Cup')
print('='*80)

print('\n1. ACTUAL WINNERS (Horses 14, 20, 7, 21):')
print('-'*80)
winners_df = df[df['is_winner']].copy()
for idx, row in winners_df.iterrows():
    avg_time = np.mean(row['recent_race_times'])
    print(f"Horse {int(row['horse_number']):2d}: {row['horse_name']:20s} | Odds: ${row['odds']:6.2f} | Weight: {row['weight']:5.1f}kg | Age: {int(row['age'])} | Barrier: {int(row['barrier']):2d} | Avg Time: {avg_time:.2f}s | Wins: {int(row['wins']):2d} | Places: {int(row['places']):2d}")

print('\n2. WINNERS STATISTICS:')
print('-'*80)
print(f"  Average Odds: ${winners_df['odds'].mean():.2f} (Range: ${winners_df['odds'].min():.2f} - ${winners_df['odds'].max():.2f})")
print(f"  Average Weight: {winners_df['weight'].mean():.1f}kg (Range: {winners_df['weight'].min():.1f} - {winners_df['weight'].max():.1f}kg)")
print(f"  Average Age: {winners_df['age'].mean():.1f} years (Range: {int(winners_df['age'].min())} - {int(winners_df['age'].max())})")
print(f"  Average Barrier: {winners_df['barrier'].mean():.1f} (Range: {int(winners_df['barrier'].min())} - {int(winners_df['barrier'].max())})")
avg_times = [np.mean(row['recent_race_times']) for _, row in winners_df.iterrows()]
print(f"  Average Recent Race Time: {np.mean(avg_times):.2f}s")
print(f"  Average Wins: {winners_df['wins'].mean():.1f}")
print(f"  Average Places: {winners_df['places'].mean():.1f}")

print('\n3. ALL HORSES STATISTICS:')
print('-'*80)
print(f"  Average Odds: ${df['odds'].mean():.2f} (Range: ${df['odds'].min():.2f} - ${df['odds'].max():.2f})")
print(f"  Average Weight: {df['weight'].mean():.1f}kg (Range: {df['weight'].min():.1f} - {df['weight'].max():.1f}kg)")
print(f"  Average Age: {df['age'].mean():.1f} years")
print(f"  Average Barrier: {df['barrier'].mean():.1f}")

print('\n4. KEY PATTERNS:')
print('-'*80)
print('Weight Analysis:')
light_horses = df[df['weight'] <= 54.5]
winners_in_light = len(light_horses[light_horses['is_winner']])
print(f"  Horses ≤54.5kg: {len(light_horses)} horses, {winners_in_light} winners ({winners_in_light/len(light_horses)*100:.1f}%)")
print(f"  All 4 winners were ≤54.5kg!")

print('\nAge Analysis:')
age_range = df[(df['age'] >= 4) & (df['age'] <= 7)]
winners_in_age = len(age_range[age_range['is_winner']])
print(f"  Horses age 4-7: {len(age_range)} horses, {winners_in_age} winners ({winners_in_age/len(age_range)*100:.1f}%)")
print(f"  All 4 winners were age 4-7!")

print('\nOdds Analysis:')
favorites = df[df['odds'] <= 10]
longshots = df[df['odds'] > 30]
winners_fav = len(favorites[favorites['is_winner']])
winners_long = len(longshots[longshots['is_winner']])
print(f"  Favorites (≤$10): {len(favorites)} horses, {winners_fav} winners ({winners_fav/len(favorites)*100:.1f}%)")
print(f"  Longshots (>$30): {len(longshots)} horses, {winners_long} winners ({winners_long/len(longshots)*100:.1f}%)")

print('\n5. FEATURE CORRELATIONS WITH WINNING:')
print('-'*80)
df['avg_recent_time'] = df['recent_race_times'].apply(np.mean)
df['best_recent_time'] = df['recent_race_times'].apply(np.min)
print(f"  Weight correlation: {df['weight'].corr(df['is_winner']):.4f} (should be negative)")
print(f"  Age correlation: {df['age'].corr(df['is_winner']):.4f}")
print(f"  Odds correlation: {df['odds'].corr(df['is_winner']):.4f} (should be negative)")
print(f"  Avg recent time correlation: {df['avg_recent_time'].corr(df['is_winner']):.4f} (should be negative - faster is better)")
print(f"  Best recent time correlation: {df['best_recent_time'].corr(df['is_winner']):.4f} (should be negative)")
print(f"  Wins correlation: {df['wins'].corr(df['is_winner']):.4f}")
print(f"  Places correlation: {df['places'].corr(df['is_winner']):.4f}")

print('\n6. RECENT RACE TIMES ANALYSIS:')
print('-'*80)
df['consistency'] = df['recent_race_times'].apply(lambda x: np.std(x))

print('Winners recent race performance:')
winners_df = df[df['is_winner']].copy()
for idx, row in winners_df.iterrows():
    print(f"  Horse {int(row['horse_number']):2d}: Avg={row['avg_recent_time']:.2f}s, Best={row['best_recent_time']:.2f}s, Consistency={row['consistency']:.2f}")

print(f"\nWinners avg: {winners_df['avg_recent_time'].mean():.2f}s (All horses: {df['avg_recent_time'].mean():.2f}s)")
print(f"Winners best: {winners_df['best_recent_time'].mean():.2f}s (All horses: {df['best_recent_time'].mean():.2f}s)")

print('\n7. TOP PERFORMERS BY METRIC:')
print('-'*80)
print('Lightest horses (≤53kg):')
light = df[df['weight'] <= 53.0].sort_values('weight')
for idx, row in light.head(8).iterrows():
    marker = '✓' if row['is_winner'] else ' '
    print(f"  {marker} Horse {int(row['horse_number']):2d}: {row['weight']:.1f}kg | Odds: ${row['odds']:.2f}")

print('\nFastest average recent times:')
fast = df.nsmallest(8, 'avg_recent_time')
for idx, row in fast.iterrows():
    marker = '✓' if row['is_winner'] else ' '
    print(f"  {marker} Horse {int(row['horse_number']):2d}: {row['avg_recent_time']:.2f}s | Weight: {row['weight']:.1f}kg | Odds: ${row['odds']:.2f}")

print('\n8. UNIQUE WINNER CHARACTERISTICS:')
print('-'*80)
non_winners = df[~df['is_winner']]
weight_diff = winners_df['weight'].mean() - non_winners['weight'].mean()
time_diff = winners_df['avg_recent_time'].mean() - non_winners['avg_recent_time'].mean()
print(f"  Weight: Winners avg {winners_df['weight'].mean():.1f}kg vs Non-winners {non_winners['weight'].mean():.1f}kg (diff: {weight_diff:.1f}kg lighter)")
print(f"  Avg Recent Time: Winners {winners_df['avg_recent_time'].mean():.2f}s vs Non-winners {non_winners['avg_recent_time'].mean():.2f}s (diff: {time_diff:.2f}s faster)")

print('\n9. SPECIFIC INSIGHTS FOR HORSES 20, 7, 21:')
print('-'*80)
for horse_num in [20, 7, 21]:
    row = df[df['horse_number'] == horse_num].iloc[0]
    weight_rank = df['weight'].rank(ascending=True)[df['horse_number']==horse_num].iloc[0]
    odds_rank = df['odds'].rank(ascending=True)[df['horse_number']==horse_num].iloc[0]
    age_rank = df['age'].rank(ascending=False)[df['horse_number']==horse_num].iloc[0]
    time_rank = df['avg_recent_time'].rank(ascending=True)[df['horse_number']==horse_num].iloc[0]
    best_rank = df['best_recent_time'].rank(ascending=True)[df['horse_number']==horse_num].iloc[0]
    
    print(f"\nHorse {horse_num} ({row['horse_name']}):")
    print(f"  Weight: {row['weight']:.1f}kg (Rank: {weight_rank:.0f}/{len(df)} - lighter is better)")
    print(f"  Odds: ${row['odds']:.2f} (Rank: {odds_rank:.0f}/{len(df)} - lower is better)")
    print(f"  Age: {int(row['age'])} (Rank: {age_rank:.0f}/{len(df)})")
    print(f"  Avg Recent Time: {row['avg_recent_time']:.2f}s (Rank: {time_rank:.0f}/{len(df)} - faster is better)")
    print(f"  Best Recent Time: {row['best_recent_time']:.2f}s (Rank: {best_rank:.0f}/{len(df)} - faster is better)")
    print(f"  Wins: {int(row['wins'])} | Places: {int(row['places'])}")
    if row['weight'] <= 52:
        strength = "Very light weight"
    elif row['weight'] <= 54.5:
        strength = "Light weight"
    else:
        strength = ""
    print(f"  Key Strength: {strength}")

