"""
CLI interface for Melbourne Cup predictions.

Provides a command-line interface to display top-3 predictions
in a formatted table.
"""

import argparse
import sys
import os
from datetime import datetime
from typing import Optional

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import predict_melbourne_cup, PredictionEngine, load_input_data
import os


def format_table(predictions_df, top_n: int = 3) -> str:
    """
    Format predictions as a table.
    
    Args:
        predictions_df: DataFrame with predictions
        top_n: Number of top predictions to display
        
    Returns:
        Formatted table string
    """
    top_predictions = predictions_df.head(top_n)
    
    # Create table header
    header = "Horse Number | Horse Name           | Predicted Probability (Top 3)"
    separator = "-" * len(header)
    
    # Format rows
    rows = []
    for _, row in top_predictions.iterrows():
        horse_num = str(int(row['horse_number']))
        horse_name = row['horse_name'][:20]  # Truncate long names
        probability = f"{row['top3_probability']:.2f}"
        
        # Format with proper spacing
        row_str = f"{horse_num:<13} | {horse_name:<20} | {probability}"
        rows.append(row_str)
    
    # Combine header, separator, and rows
    table = "\n".join([header, separator] + rows)
    
    return table


def display_csv(predictions_df):
    """
    Display predictions in CSV format.
    
    Args:
        predictions_df: DataFrame with predictions (already sorted by probability)
    """
    # Ensure sorted by probability descending
    predictions_df = predictions_df.sort_values('top3_probability', ascending=False).reset_index(drop=True)
    
    # Print CSV header
    print("\nCSV Format (ordered by prediction probability):")
    print("-" * 70)
    print("Horse Number,Horse Name,Predicted Probability (Top 3)")
    
    # Print CSV rows
    for _, row in predictions_df.iterrows():
        horse_num = int(row['horse_number'])
        horse_name = row['horse_name']
        probability = f"{row['top3_probability']:.4f}"
        print(f"{horse_num},{horse_name},{probability}")
    
    print("-" * 70)


def display_predictions(predictions_df, top_n: int = 3, verbose: bool = False, csv_format: bool = False):
    """
    Display predictions in formatted table and/or CSV format.
    
    Args:
        predictions_df: DataFrame with predictions
        top_n: Number of top predictions to display in table
        verbose: Whether to show additional information
        csv_format: Whether to also display in CSV format
    """
    # Ensure sorted by probability descending
    predictions_df = predictions_df.sort_values('top3_probability', ascending=False).reset_index(drop=True)
    
    if csv_format:
        # Display CSV format first
        display_csv(predictions_df)
    
    print("\n" + "=" * 70)
    print("MELBOURNE CUP TOP-3 PREDICTIONS")
    print("=" * 70)
    
    # Display top predictions
    print(format_table(predictions_df, top_n=top_n))
    
    if verbose:
        print("\n" + "-" * 70)
        print("Full Predictions:")
        print("-" * 70)
        print(predictions_df[['horse_number', 'horse_name', 'top3_probability']].to_string(index=False))
    
    print("\n" + "=" * 70)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Predict top-3 finishers for the Melbourne Cup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/cli.py --year 2024
  python src/cli.py --input data/processed/2024_lineup.csv --output results.csv
  python src/cli.py --input data/processed/2024_lineup.csv --verbose
        """
    )
    
    parser.add_argument(
        '--year',
        type=int,
        default=datetime.now().year,
        help='Year of Melbourne Cup (default: current year)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to input data file (CSV/JSON). If not provided, mock data will be generated.'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results.csv',
        help='Path to save results CSV (default: results.csv)'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing trained models (default: models)'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=3,
        help='Number of top predictions to display (default: 3)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Display full predictions table'
    )
    
    parser.add_argument(
        '--no-randomness',
        action='store_true',
        help='Disable random variation in predictions (use deterministic model output)'
    )
    
    parser.add_argument(
        '--randomness-factor',
        type=float,
        default=0.01,
        help='Factor controlling randomness (0.0-1.0, default: 0.01 = 1% subtle variation)'
    )
    
    parser.add_argument(
        '--csv',
        action='store_true',
        help='Display predictions in CSV format'
    )
    
    args = parser.parse_args()
    
    try:
        # Check if model exists
        model_path = os.path.join(args.model_dir, "logistic_model.pkl")
        models_exist = os.path.exists(model_path)
        
        if not models_exist:
            print("Warning: Trained models not found.")
            print("Please train models first by running: python src/train_models.py")
            print("\nProceeding with mock data generation for demonstration...")
            
            # Load input data
            input_data = load_input_data(args.input, args.year)
            print(f"\nLoaded {len(input_data)} horses for year {args.year}")
            print("\nNote: To make actual predictions, please train the models first.")
            print("The system will generate mock predictions for demonstration purposes.")
            
            # Save input data for reference
            if args.input is None:
                os.makedirs("data/processed", exist_ok=True)
                input_path = f"data/processed/{args.year}_lineup.csv"
                input_data.to_csv(input_path, index=False)
                print(f"\nInput data saved to {input_path}")
            
            return
        
        # Make predictions
        print(f"Loading models from {args.model_dir}...")
        predictions = predict_melbourne_cup(
            input_path=args.input,
            year=args.year,
            model_dir=args.model_dir,
            output_path=args.output,
            add_randomness=not args.no_randomness,
            randomness_factor=args.randomness_factor
        )
        
        # Display results
        display_predictions(predictions, top_n=args.top_n, verbose=args.verbose, csv_format=args.csv)
        
        print(f"\nResults saved to: {args.output}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure:")
        print("1. Models are trained: python src/train_models.py")
        print("2. Input data file exists (if using --input)")
        print("3. Model directory is correct (if using --model-dir)")
        sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

