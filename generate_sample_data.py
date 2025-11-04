"""
Utility script to generate sample data files for testing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_fetch import DataFetcher
import pandas as pd
import numpy as np

def generate_sample_historical_data(years=[2020, 2021, 2022, 2023], output_dir="data/historical"):
    """Generate sample historical data files."""
    fetcher = DataFetcher()
    os.makedirs(output_dir, exist_ok=True)
    
    for year in years:
        print(f"Generating sample data for {year}...")
        df = fetcher.generate_mock_data(num_horses=24, year=year)
        # Add finishing positions
        np.random.seed(year)  # For reproducibility
        df['finishing_position'] = np.random.permutation(range(1, len(df) + 1))
        
        # Save to file
        output_path = os.path.join(output_dir, f"melbourne_cup_{year}.csv")
        df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")
    
    print(f"\nSample historical data generated for years: {years}")
    print(f"Files saved in: {output_dir}")

def generate_sample_prediction_data(year=2024, output_dir="data/processed"):
    """Generate sample prediction data (no finishing positions)."""
    fetcher = DataFetcher()
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating sample prediction data for {year}...")
    df = fetcher.generate_mock_data(num_horses=24, year=year)
    # Don't include finishing_position for prediction data
    
    output_path = os.path.join(output_dir, f"{year}_lineup.csv")
    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"\nSample prediction data generated for year: {year}")
    print(f"File saved in: {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample data files')
    parser.add_argument('--historical', action='store_true', 
                       help='Generate historical training data')
    parser.add_argument('--prediction', action='store_true',
                       help='Generate prediction data (current year lineup)')
    parser.add_argument('--years', nargs='+', type=int, 
                       default=[2020, 2021, 2022, 2023],
                       help='Years for historical data')
    parser.add_argument('--year', type=int, default=2024,
                       help='Year for prediction data')
    
    args = parser.parse_args()
    
    if args.historical:
        generate_sample_historical_data(years=args.years)
    
    if args.prediction:
        generate_sample_prediction_data(year=args.year)
    
    if not args.historical and not args.prediction:
        # Generate both by default
        print("Generating both historical and prediction data...\n")
        generate_sample_historical_data(years=args.years)
        generate_sample_prediction_data(year=args.year)

