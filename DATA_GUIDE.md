# Data Guide for Melbourne Cup Prediction System

This guide explains how to obtain and prepare data for the Melbourne Cup prediction system.

## Quick Start: Using Mock Data

The system automatically generates mock data if no historical data is found. **You can start training immediately:**

```bash
python src/train_models.py
```

This will generate mock historical data for years 2020-2023 and train the models.

## Option 1: Manual Data Preparation (Recommended)

### Historical Data Format

Place CSV files in `data/historical/` with the naming pattern: `melbourne_cup_YYYY.csv`

**Required columns:**
- `year`: Year of the race (e.g., 2023)
- `horse_number`: Horse number (1-24)
- `horse_name`: Name of the horse
- `age`: Age of the horse (integer)
- `weight`: Weight carried in kg (float)
- `trainer`: Trainer name (string)
- `jockey`: Jockey name (string)
- `barrier`: Barrier position (1-24)
- `track_condition`: Track condition (e.g., "Good", "Soft", "Heavy", "Firm")
- `distance`: Race distance in meters (typically 3200)
- `wins`: Career wins (integer)
- `places`: Career places (integer)
- `recent_race_times`: Recent race times as JSON array string (e.g., "[120.5, 121.2, 119.8]")
- `odds`: Starting odds (float, optional)
- `finishing_position`: Final finishing position (1-24) - **REQUIRED for training data**

### Sample Data File Structure

Example: `data/historical/melbourne_cup_2023.csv`

```csv
year,horse_number,horse_name,age,weight,trainer,jockey,barrier,track_condition,distance,wins,places,recent_race_times,odds,finishing_position
2023,1,Without A Fight,6,57.5,A. Freedman,M. Zahra,1,Good,3200,8,12,"[120.5, 121.2, 119.8, 120.1, 119.5]",8.5,1
2023,2,Soulcombe,5,56.0,C. Waller,J. McDonald,5,Good,3200,6,8,"[121.0, 120.8, 121.5, 120.2, 119.9]",12.0,3
2023,3,Serpentine,6,55.5,G. Waterhouse,D. Lane,12,Good,3200,5,10,"[122.1, 121.8, 122.5, 121.0, 120.5]",15.0,5
...
```

### Where to Find Real Data

**Official Sources:**
1. **Racing.com**: https://www.racing.com - Provides race results and statistics
2. **Racing Victoria**: https://www.racingvictoria.com.au - Official racing authority
3. **Punters.com.au**: https://www.punters.com.au - Racing data and statistics

**Data Aggregation Sites:**
1. **Kaggle**: Search for "Melbourne Cup" or "horse racing" datasets
2. **GitHub**: Search for horse racing data repositories
3. **Open Data Portals**: Some racing authorities publish open data

### Creating Your Own Data File

1. **Collect race information** for each Melbourne Cup year:
   - Race results (finishing positions)
   - Horse details (age, weight, trainer, jockey)
   - Race conditions (barrier, track condition)
   - Performance statistics (wins, places, recent times)

2. **Format as CSV** following the structure above

3. **Save to** `data/historical/melbourne_cup_YYYY.csv`

4. **For prediction data** (current year lineup), save to `data/processed/YYYY_lineup.csv` (without `finishing_position` column)

## Option 2: Using Kaggle Datasets

If you find a Melbourne Cup dataset on Kaggle:

### Setup Kaggle API

1. **Get your API credentials:**
   - Go to https://www.kaggle.com/account
   - Scroll to "API" section
   - Click "Create New Token" to download `kaggle.json`

2. **Install Kaggle package:**
   ```bash
   pip install kaggle
   ```

3. **Place credentials:**
   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Using Kaggle Data

Create a script to download and format Kaggle data:

```python
from src.data_fetch import DataFetcher

fetcher = DataFetcher()
# Example: download from a Kaggle dataset
df = fetcher.fetch_from_kaggle("username/dataset-name", "melbourne_cup_data.csv")

# Format and save to historical directory
if df is not None:
    df.to_csv("data/historical/melbourne_cup_2023.csv", index=False)
```

## Option 3: Web Scraping (Advanced)

The system includes scaffolding for web scraping. To implement:

1. **Identify data source** (e.g., Racing.com race results page)
2. **Modify `scrape_racing_data()` in `src/data_fetch.py`**
3. **Parse HTML structure** using BeautifulSoup
4. **Extract and format data** into required structure

**Important:** Always check website terms of service and robots.txt before scraping.

## Data Requirements Summary

### For Training (Historical Data)
- **Location**: `data/historical/melbourne_cup_YYYY.csv`
- **Required**: All columns including `finishing_position`
- **Multiple years**: Include multiple CSV files for better training

### For Prediction (Current Year)
- **Location**: `data/processed/YYYY_lineup.csv` or passed via `--input`
- **Required**: All columns EXCEPT `finishing_position`
- **Format**: Same as training data but without outcome

## Example: Generate Sample Data

You can generate a sample file to see the format:

```python
from src.data_fetch import DataFetcher
import pandas as pd

fetcher = DataFetcher()
# Generate mock data for 2023
df = fetcher.generate_mock_data(num_horses=24, year=2023)
# Add finishing positions
df['finishing_position'] = range(1, 25)
# Save to historical directory
df.to_csv('data/historical/melbourne_cup_2023.csv', index=False)
print("Sample data saved!")
```

## Verifying Your Data

Before training, verify your data format:

```python
import pandas as pd

df = pd.read_csv('data/historical/melbourne_cup_2023.csv')
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("Missing values:", df.isnull().sum())
print("\nFirst few rows:")
print(df.head())
```

## Tips

1. **More data = better models**: Include multiple years (2015-2023) for better training
2. **Clean data**: Ensure `finishing_position` values are integers 1-24
3. **Recent times**: Format as JSON array string: `"[120.5, 121.2, 119.8]"`
4. **Track conditions**: Use standard values: "Good", "Soft", "Heavy", "Firm"
5. **Barrier numbers**: Must be unique per race (1-24)

## Need Help?

If you're having trouble finding data:
- Start with mock data to test the system
- Search Kaggle for "horse racing" or "Melbourne Cup" datasets
- Check Racing.com archives for historical results
- Consider using web scraping tools (respectfully and legally)

