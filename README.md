# Melbourne Cup Top-3 Prediction

Predicts each runner's chance of finishing in the top three of the Melbourne Cup.

## Current pipeline

- **Data:** `data/verified/` holds Wikipedia-checked results (barrier, weight, finishing order). `data/market/cup_bsp.csv` holds Betfair Starting Prices for the Cups since 2020. The old `data/historical/` CSVs are kept only for the comparison in `docs/verification_report.md`.
- **Model:** a regularised Plackett-Luce (conditional logit) fitted on the first three finishers of each race, with exact top-3 probabilities.
- **Benchmark:** the Betfair market's normalised implied probabilities.
- **Evaluation:** leave-one-year-out, per-race log loss, race-level bootstrap. `docs/evaluation_report.md` gives the verdict: on the 2020-2025 market years the combined model did not beat the market (top-3 log loss +0.380 nats, 95% CI includes 0), and over those six Cups the market could not be told apart from uniform guessing (winner log loss 3.365 vs 3.135, top-3 9.295 vs 9.270, CI for the top-3 difference [-1.521, +1.538]), while the weight/barrier model and the combined model scored worse than uniform on top-3 log loss (9.583 and 9.675 vs 9.270). The fixed rule therefore uses market-only probabilities, so predictions carry no information beyond the odds supplied. Nothing here shows a betting edge.
- **Frozen predictions:** `predictions/` holds predictions committed and tagged before a race (the folder is created when the first prediction is frozen).

### Commands

    python -m pytest                              # all tests
    python scripts/fetch_cup_bsp.py               # download Cup BSP from Betfair
    python scripts/compare_historical.py          # check verified files, compare with old CSVs
    python -m src.evaluate                        # leave-one-year-out report
    python -m src.predict_cup data/processed/2026_lineup_provisional.csv
    python -m src.strategies                      # replay bet shapes on past Cups (docs/strategy_report.md)

Run all commands from the repository root. `data/processed/` is gitignored, so the lineup CSV is a local file; its `odds` column must hold decimal win odds from the bookmaker you will bet with (the model's market features were trained on Betfair BSP).

## Legacy (do not use for decisions)

`src/train_models.py`, `src/predict.py`, `src/advanced_models.py`, `src/predict_advanced.py` and `src/cli.py` are the original XGBoost / LSTM / logistic pipeline. They were trained on partly synthetic data, scored finishing position instead of each race's field, and were tuned toward the 2025 result. The legacy model ranked the 2025 winner, Half Yours, 14th (see `analyze_failure.py`), and its held-out top-3 accuracy was 0.0 (`models/evaluation_results.json`). They are kept for reference only.

## Legacy pipeline reference

Everything below describes the legacy pipeline (mock data, XGBoost/LSTM, MAE/RMSE) and is not used for current predictions. Use the commands in "Current pipeline" above.

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

## Getting Data

The system has three ways to get data:

1. **Mock Data (Automatic)**: The system generates mock data automatically if no historical data is found. You can start training immediately!

2. **Manual Preparation**: Place CSV files in `data/historical/` named `melbourne_cup_YYYY.csv` (see `DATA_GUIDE.md` for format details)

3. **Generate Sample Data**: Run `python generate_sample_data.py` to create sample files

See `DATA_GUIDE.md` for detailed instructions on data sources, formats, and preparation.

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
