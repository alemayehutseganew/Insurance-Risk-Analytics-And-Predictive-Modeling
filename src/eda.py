"""Helper utilities for Exploratory Data Analysis (EDA).

Place common EDA functions here to import from notebooks or scripts.
"""
from typing import Tuple
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV safely and parse dates if present.

    Args:
        path: path to CSV file

    Returns:
        DataFrame
    """
    df = pd.read_csv(path)
    return df


def basic_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return descriptive stats and missing-value counts."""
    desc = df.describe(include='all')
    missing = df.isnull().sum()
    return desc, missing


def loss_ratio(df: pd.DataFrame) -> float:
    """Compute portfolio loss ratio defined as TotalClaims / TotalPremium."""
    if 'TotalClaims' not in df.columns or 'TotalPremium' not in df.columns:
        raise KeyError('DataFrame must contain TotalClaims and TotalPremium columns')
    total_claims = df['TotalClaims'].sum()
    total_premium = df['TotalPremium'].sum()
    return float(total_claims) / float(total_premium) if total_premium != 0 else float('nan')
