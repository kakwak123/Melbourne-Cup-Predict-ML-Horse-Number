# Melbourne Cup Top-3 Prediction System

An AI-powered system that predicts the top 3 finishing horses for the Melbourne Cup using machine learning models (XGBoost regression and LSTM neural networks).

## Features

- **Data Pipeline**: Automated data collection, cleaning, and preprocessing from public sources
- **Dual Model Architecture**: 
  - XGBoost regression for finishing position prediction
  - LSTM neural network for capturing temporal patterns in horse performance
- **Ensemble Prediction**: Combines both models for improved accuracy
- **Evaluation Metrics**: MAE, RMSE, and top-3 ranking accuracy
- **CLI Interface**: Simple command-line tool for predictions

## Project Structure

```
melbourne-cup-predict/
├── data/
│   ├── raw/              # Raw scraped/downloaded data
│   ├── processed/        # Cleaned and preprocessed data
│   └── historical/       # Historical Melbourne Cup data
├── models/               # Saved trained models
├── src/
│   ├── data_fetch.py     # Data collection and cleaning
│   ├── train_models.py   # Model training and evaluation
│   ├── predict.py        # Prediction logic
│   └── cli.py            # CLI interface
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Melbourne-Cup-Predict-ML-Horse-Number
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Train models** (uses mock historical data if none provided):
```bash
python src/train_models.py
```

3. **Make predictions:**
```bash
python src/cli.py --year 2024
```

### Training Models

First, ensure you have historical Melbourne Cup data in `data/historical/`. The system will generate mock historical data if none is available. Then train the models:

```bash
python src/train_models.py
```

This will:
- Load and preprocess historical data
- Train XGBoost regression model with hyperparameter tuning
- Train LSTM neural network for temporal patterns
- Evaluate both models and save them to `models/` directory

### Making Predictions

For a specific year's Melbourne Cup:

```bash
python src/cli.py --year 2024 --input data/processed/2024_lineup.csv
```

Or use default settings (generates mock data if no input provided):

```bash
python src/cli.py
```

### Command-Line Options

**CLI Options (`src/cli.py`):**
- `--year`: Year of Melbourne Cup (default: current year)
- `--input`: Path to input data file (CSV/JSON format)
- `--output`: Path to save results CSV (default: results.csv)
- `--model-dir`: Directory containing trained models (default: models)
- `--top-n`: Number of top predictions to display (default: 3)
- `--verbose`: Display full predictions table

**Prediction Script (`src/predict.py`):**
- `--input`: Path to input data file
- `--year`: Year of Melbourne Cup
- `--output`: Path to save results CSV
- `--model-dir`: Directory containing trained models

## Input Data Format

The input CSV/JSON should contain the following columns:

- `horse_number`: Horse number (1-24)
- `horse_name`: Name of the horse
- `age`: Age of the horse
- `weight`: Weight carried (kg)
- `trainer`: Trainer name
- `jockey`: Jockey name
- `barrier`: Barrier position (1-24)
- `track_condition`: Track condition (e.g., "Good", "Soft")
- `distance`: Race distance (typically 3200m for Melbourne Cup)
- `wins`: Career wins
- `places`: Career places
- `recent_race_times`: Recent race times (comma-separated or JSON)
- `odds`: Starting odds (optional)

## Output Format

The system outputs a formatted table:

```
Horse Number | Horse Name     | Predicted Probability (Top 3)
---------------------------------------------------------
3            | Without A Fight | 0.72
5            | Soulcombe       | 0.65
12           | Serpentine      | 0.58
```

Results are also saved to `results.csv` by default.

## Model Details

### XGBoost Regression Model
- Predicts finishing position as a continuous value (1-24)
- Uses feature importance analysis
- Hyperparameter tuning via GridSearchCV
- Default ensemble weight: 0.6

### LSTM Neural Network
- Captures temporal trends from historical performance
- Handles variable-length sequences with padding
- Architecture: Masking → LSTM(64) → LSTM(32) → Dense(16) → Dense(1)
- Sequence length: 5 races per horse
- Default ensemble weight: 0.4

### Ensemble Approach
- Combines predictions from both models using weighted average
- Converts finishing positions to top-3 probabilities using softmax transformation
- Sorts horses by probability descending

## Code Structure

- `src/data_fetch.py`: Data collection, cleaning, and preprocessing
- `src/train_models.py`: Model training and evaluation
- `src/predict.py`: Prediction engine with ensemble logic
- `src/cli.py`: Command-line interface for predictions

## Evaluation Metrics

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **Top-3 Accuracy**: Percentage of correctly predicted top-3 finishers

## License

This project is for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

