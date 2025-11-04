"""
Data collection, cleaning, and preprocessing module for Melbourne Cup predictions.

This module handles:
- Fetching data from public sources (Kaggle, web scraping, mock data)
- Cleaning and preprocessing raw data
- Feature engineering and encoding
- Data persistence
"""

import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, List, Tuple
import json
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

warnings.filterwarnings('ignore')


class DataFetcher:
    """Handles data collection from various sources."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the DataFetcher.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.historical_dir = os.path.join(data_dir, "historical")
        
        # Create directories if they don't exist
        for directory in [self.raw_dir, self.processed_dir, self.historical_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def fetch_from_kaggle(self, dataset_name: str, file_name: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from Kaggle (requires Kaggle API credentials).
        
        Args:
            dataset_name: Kaggle dataset name (format: username/dataset)
            file_name: Name of the CSV file to download
            
        Returns:
            DataFrame or None if fetch fails
        """
        try:
            import kaggle
            kaggle.api.dataset_download_file(
                dataset_name,
                file_name=file_name,
                path=self.raw_dir,
                unzip=True
            )
            file_path = os.path.join(self.raw_dir, file_name)
            if os.path.exists(file_path):
                return pd.read_csv(file_path)
        except Exception as e:
            print(f"Kaggle fetch failed: {e}")
            print("Note: Kaggle API requires credentials. Using mock data instead.")
        return None
    
    def scrape_racing_data(self, url: str, year: int) -> Optional[pd.DataFrame]:
        """
        Scrape horse racing data from Racing.com or similar sites.
        
        Args:
            url: URL to scrape
            year: Year of the race
            
        Returns:
            DataFrame or None if scrape fails
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            # Placeholder for actual scraping logic
            # This would need to be customized based on the actual website structure
            
            print(f"Scraping from {url} not fully implemented. Using mock data.")
            return None
        except Exception as e:
            print(f"Scraping failed: {e}")
            return None
    
    def generate_mock_data(self, num_horses: int = 24, year: int = 2024) -> pd.DataFrame:
        """
        Generate mock Melbourne Cup data for testing.
        
        Args:
            num_horses: Number of horses in the race
            year: Year of the race
            
        Returns:
            DataFrame with mock race data
        """
        np.random.seed(42)
        
        # Sample horse names
        horse_names = [
            "Without A Fight", "Soulcombe", "Serpentine", "Vauban", "Gold Trip",
            "Ashrun", "Breakup", "Future History", "Military Mission", "Right You Are",
            "Vow And Declare", "Absurde", "Young Werther", "Magical Lagoon", "Lastotchka",
            "True Marvel", "Okita Soushi", "Kalapour", "More Felons", "Daqiansweet Junior",
            "Point King", "Interpretation", "Just Fine", "Alenquer"
        ]
        
        # Sample trainers and jockeys
        trainers = [
            "A. Freedman", "C. Waller", "G. Waterhouse", "J. O'Brien", "M. Zahra",
            "D. O'Brien", "C. Maher", "T. Dabernig", "J. Cummings", "M. Ellerton"
        ]
        
        jockeys = [
            "M. Zahra", "J. McDonald", "D. Lane", "B. Melham", "J. Kah",
            "C. Williams", "D. Oliver", "L. Currie", "M. Dee", "D. Moor"
        ]
        
        track_conditions = ["Good", "Soft", "Heavy", "Firm"]
        
        data = []
        for i in range(num_horses):
            horse_num = i + 1
            age = np.random.randint(3, 8)
            weight = np.random.uniform(52.0, 59.0)
            barrier = np.random.randint(1, num_horses + 1)
            
            # Ensure unique barriers
            used_barriers = [d['barrier'] for d in data]
            while barrier in used_barriers:
                barrier = np.random.randint(1, num_horses + 1)
            
            wins = np.random.randint(0, 20)
            places = np.random.randint(0, 30)
            recent_times = [np.random.uniform(120.0, 140.0) for _ in range(5)]
            odds = np.random.uniform(5.0, 50.0)
            
            data.append({
                'year': year,
                'horse_number': horse_num,
                'horse_name': horse_names[i % len(horse_names)] if i < len(horse_names) else f"Horse {horse_num}",
                'age': age,
                'weight': round(weight, 1),
                'trainer': np.random.choice(trainers),
                'jockey': np.random.choice(jockeys),
                'barrier': barrier,
                'track_condition': np.random.choice(track_conditions),
                'distance': 3200,
                'wins': wins,
                'places': places,
                'recent_race_times': json.dumps(recent_times),
                'odds': round(odds, 2),
                'finishing_position': None  # Will be filled for historical data
            })
        
        return pd.DataFrame(data)
    
    def load_historical_data(self, year: Optional[int] = None) -> pd.DataFrame:
        """
        Load historical Melbourne Cup data.
        
        Args:
            year: Specific year to load, or None for all years
            
        Returns:
            DataFrame with historical data
        """
        historical_files = []
        if year:
            file_path = os.path.join(self.historical_dir, f"melbourne_cup_{year}.csv")
            if os.path.exists(file_path):
                historical_files.append(file_path)
        else:
            # Load all historical files
            for file in os.listdir(self.historical_dir):
                if file.startswith("melbourne_cup_") and file.endswith(".csv"):
                    historical_files.append(os.path.join(self.historical_dir, file))
        
        if not historical_files:
            print("No historical data found. Generating mock historical data for training.")
            # Generate mock historical data for multiple years
            all_data = []
            for y in range(2020, 2024):
                df = self.generate_mock_data(num_horses=24, year=y)
                # Add finishing positions (simulated)
                df['finishing_position'] = np.random.permutation(range(1, len(df) + 1))
                all_data.append(df)
            return pd.concat(all_data, ignore_index=True)
        
        dfs = []
        for file_path in historical_files:
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()


class DataPreprocessor:
    """Handles data cleaning and preprocessing."""
    
    def __init__(self, scaler_path: Optional[str] = None, encoder_path: Optional[str] = None):
        """
        Initialize the DataPreprocessor.
        
        Args:
            scaler_path: Path to save/load StandardScaler
            encoder_path: Path to save/load LabelEncoders
        """
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.scaler_path = scaler_path or "models/scaler.pkl"
        self.encoder_path = encoder_path or "models/encoders.pkl"
        self.feature_columns = None
        self.is_fitted = False
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data by handling missing values and data types.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Parse recent_race_times if it's a string
        if 'recent_race_times' in df.columns:
            if df['recent_race_times'].dtype == 'object':
                df['recent_race_times'] = df['recent_race_times'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
        
        # Calculate derived features
        df['career_starts'] = df.get('wins', 0) + df.get('places', 0) + np.random.randint(0, 20)
        df['win_rate'] = df['wins'] / (df['career_starts'] + 1)
        df['place_rate'] = df['places'] / (df['career_starts'] + 1)
        
        # Extract recent race time statistics
        if 'recent_race_times' in df.columns:
            df['avg_recent_time'] = df['recent_race_times'].apply(
                lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else np.nan
            )
            df['best_recent_time'] = df['recent_race_times'].apply(
                lambda x: np.min(x) if isinstance(x, list) and len(x) > 0 else np.nan
            )
            df['num_recent_races'] = df['recent_race_times'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        else:
            df['avg_recent_time'] = np.nan
            df['best_recent_time'] = np.nan
            df['num_recent_races'] = 0
        
        # Handle missing values in numerical columns
        numerical_cols = ['age', 'weight', 'barrier', 'wins', 'places', 'odds',
                         'career_starts', 'win_rate', 'place_rate', 
                         'avg_recent_time', 'best_recent_time']
        
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].median(), inplace=True)
        
        # Ensure categorical columns are strings
        categorical_cols = ['trainer', 'jockey', 'track_condition', 'horse_name']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].replace('nan', 'Unknown')
        
        return df
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: DataFrame to encode
            fit: Whether to fit encoders (True for training, False for prediction)
            
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        # Label encode categorical variables
        categorical_cols = ['trainer', 'jockey', 'track_condition']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        known_classes = set(self.label_encoders[col].classes_)
                        df[col] = df[col].apply(
                            lambda x: x if x in known_classes else 'Unknown'
                        )
                        # Add Unknown to encoder if not present
                        if 'Unknown' not in self.label_encoders[col].classes_:
                            self.label_encoders[col].classes_ = np.append(
                                self.label_encoders[col].classes_, 'Unknown'
                            )
                        df[col + '_encoded'] = self.label_encoders[col].transform(df[col])
                    else:
                        df[col + '_encoded'] = 0
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features.
        
        Args:
            df: DataFrame to normalize
            fit: Whether to fit scaler (True for training, False for prediction)
            
        Returns:
            DataFrame with normalized features
        """
        df = df.copy()
        
        # Select numerical features
        numerical_features = [
            'age', 'weight', 'barrier', 'wins', 'places', 'odds',
            'career_starts', 'win_rate', 'place_rate',
            'avg_recent_time', 'best_recent_time', 'num_recent_races'
        ]
        
        # Add encoded categorical features
        encoded_features = ['trainer_encoded', 'jockey_encoded', 'track_condition_encoded']
        
        feature_cols = [col for col in numerical_features + encoded_features if col in df.columns]
        
        if fit:
            self.feature_columns = feature_cols
            df_scaled = pd.DataFrame(
                self.scaler.fit_transform(df[feature_cols]),
                columns=feature_cols,
                index=df.index
            )
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before transform. Call with fit=True first.")
            df_scaled = pd.DataFrame(
                self.scaler.transform(df[feature_cols]),
                columns=feature_cols,
                index=df.index
            )
        
        # Replace original columns with scaled versions
        for col in feature_cols:
            df[col] = df_scaled[col]
        
        return df
    
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline: clean -> encode -> normalize.
        
        Args:
            df: Raw DataFrame
            fit: Whether to fit transformers (True for training, False for prediction)
            
        Returns:
            Preprocessed DataFrame
        """
        df = self.clean_data(df)
        df = self.encode_features(df, fit=fit)
        df = self.normalize_features(df, fit=fit)
        return df
    
    def save_preprocessors(self):
        """Save scaler and encoders to disk."""
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.label_encoders, self.encoder_path)
        print(f"Preprocessors saved to {self.scaler_path} and {self.encoder_path}")
    
    def load_preprocessors(self):
        """Load scaler and encoders from disk."""
        if os.path.exists(self.scaler_path) and os.path.exists(self.encoder_path):
            self.scaler = joblib.load(self.scaler_path)
            self.label_encoders = joblib.load(self.encoder_path)
            self.is_fitted = True
            print(f"Preprocessors loaded from {self.scaler_path} and {self.encoder_path}")
        else:
            raise FileNotFoundError("Preprocessors not found. Train models first.")


def prepare_training_data(data_dir: str = "data", save_processed: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare training data from historical records.
    
    Args:
        data_dir: Base directory for data
        save_processed: Whether to save processed data to disk
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    fetcher = DataFetcher(data_dir)
    preprocessor = DataPreprocessor()
    
    # Load historical data
    historical_df = fetcher.load_historical_data()
    
    if historical_df.empty:
        raise ValueError("No historical data available for training.")
    
    # Filter out rows without finishing positions
    training_df = historical_df[historical_df['finishing_position'].notna()].copy()
    
    if training_df.empty:
        raise ValueError("No training data with finishing positions found.")
    
    # Preprocess data
    processed_df = preprocessor.preprocess(training_df, fit=True)
    
    # Save preprocessors
    preprocessor.save_preprocessors()
    
    # Prepare features and target
    feature_cols = preprocessor.feature_columns
    X = processed_df[feature_cols]
    y = processed_df['finishing_position']
    
    # Save processed data if requested
    if save_processed:
        output_path = os.path.join(data_dir, "processed", "training_data.csv")
        processed_df.to_csv(output_path, index=False)
        print(f"Processed training data saved to {output_path}")
    
    return X, y


def prepare_prediction_data(input_path: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Prepare prediction data from input file.
    
    Args:
        input_path: Path to input CSV/JSON file
        data_dir: Base directory for data
        
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessors()
    
    # Load input data
    if input_path.endswith('.json'):
        df = pd.read_json(input_path)
    else:
        df = pd.read_csv(input_path)
    
    # Preprocess data
    processed_df = preprocessor.preprocess(df, fit=False)
    
    return processed_df


if __name__ == "__main__":
    # Example usage
    print("Generating mock data...")
    fetcher = DataFetcher()
    mock_data = fetcher.generate_mock_data(num_horses=24, year=2024)
    
    print("\nCleaning and preprocessing data...")
    preprocessor = DataPreprocessor()
    processed = preprocessor.preprocess(mock_data, fit=True)
    
    print("\nProcessed data shape:", processed.shape)
    print("\nSample processed data:")
    print(processed.head())

