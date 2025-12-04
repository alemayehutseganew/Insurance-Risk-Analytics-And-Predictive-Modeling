import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parents[1] / 'MachineLearningRating_v3'
DATA_FILE = DATA_DIR / 'MachineLearningRating_v3.txt'


def load_data(path=DATA_FILE):
    df = pd.read_csv(path, sep='|', low_memory=False, parse_dates=['TransactionMonth'],)
    return df


def preprocess_for_severity(df):
    # Filter for claims > 0
    df = df.copy()
    df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
    df_claims = df[df['HasClaim'] == 1].copy()

    # Basic feature engineering
    df_claims['VehicleAge'] = df_claims['TransactionMonth'].dt.year - pd.to_numeric(df_claims['RegistrationYear'], errors='coerce')
    df_claims['LogTotalClaims'] = np.log1p(df_claims['TotalClaims'])

    # Select a small set of features for baseline
    features = ['VehicleAge', 'Cylinders', 'cubiccapacity', 'Kilowatts', 'TermFrequency', 'CalculatedPremiumPerTerm']
    features = [f for f in features if f in df_claims.columns]

    X = df_claims[features].copy()
    y = df_claims['TotalClaims'].copy()

    # Simple imputation
    X = X.fillna(X.median())

    return X, y


def train_test_split_df(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
