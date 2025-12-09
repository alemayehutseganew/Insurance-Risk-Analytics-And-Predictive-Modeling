from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    root_mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - xgboost is optional during install
    XGBRegressor = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - xgboost is optional during install
    XGBClassifier = None

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
DATA_FILE = DATA_DIR / 'MachineLearningRating_v3.txt'


def load_data(path=DATA_FILE, *, nrows: int | None = None, usecols: list[str] | None = None):
    """Load the raw dataset from disk.

    Parameters
    ----------
    path: Path-like
        Location of the raw pipe-delimited file.
    nrows: Optional[int]
        Limit the number of rows read for faster experimentation. Defaults to reading all rows.
    usecols: Optional[List[str]]
        Restrict to a subset of columns to reduce memory usage.
    """

    df = pd.read_csv(
        path,
        sep='|',
        low_memory=False,
        parse_dates=['TransactionMonth'],
        nrows=nrows,
        usecols=usecols,
    )
    return df


NUMERIC_FEATURES = [
    'VehicleAge',
    'VehicleIntroAge',
    'PolicyTenureMonths',
    'TermFrequency',
    'TotalPremium',
    'CalculatedPremiumPerTerm',
    'SumInsured',
    'Cylinders',
    'cubiccapacity',
    'kilowatts',
]

CATEGORICAL_FEATURES = [
    'Province',
    'VehicleType',
    'make',
    'Model',
    'CoverType',
    'CoverCategory',
    'CoverGroup',
    'StatutoryRiskType',
    'Product',
    'MaritalStatus',
    'Gender',
]


def _engineer_common_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create shared engineered columns used across modeling tasks."""

    engineered = df.copy()
    engineered['HasClaim'] = (engineered['TotalClaims'] > 0).astype(int)

    def _safe_year(series):
        return pd.to_numeric(series, errors='coerce')

    engineered['VehicleAge'] = (
        engineered['TransactionMonth'].dt.year - _safe_year(engineered['RegistrationYear'])
    )
    engineered['VehicleIntroAge'] = (
        engineered['TransactionMonth'].dt.year
        - pd.to_datetime(engineered.get('VehicleIntroDate', pd.NaT), errors='coerce').dt.year
    )

    if 'PolicyID' in engineered.columns:
        first_tx = engineered.groupby('PolicyID')['TransactionMonth'].transform('min')
        engineered['PolicyTenureMonths'] = (
            (engineered['TransactionMonth'] - first_tx).dt.days / 30.0
        )
    else:
        engineered['PolicyTenureMonths'] = np.nan

    for col in ['VehicleAge', 'VehicleIntroAge']:
        engineered[col] = engineered[col].clip(lower=0, upper=40)

    return engineered


def _select_feature_columns(df: pd.DataFrame, exclude: Tuple[str, ...] = ()) -> pd.DataFrame:
    available_numeric = [feat for feat in NUMERIC_FEATURES if feat in df.columns and feat not in exclude]
    available_categorical = [
        feat for feat in CATEGORICAL_FEATURES if feat in df.columns and feat not in exclude
    ]
    features = available_numeric + available_categorical
    return df[features].copy()


def preprocess_for_severity(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target for claim severity models."""

    engineered = _engineer_common_features(df)
    df_claims = engineered[engineered['HasClaim'] == 1].copy()

    X = _select_feature_columns(df_claims)
    y = df_claims['TotalClaims'].astype(float)

    return X, y


def preprocess_for_premium(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features/target for premium prediction models."""

    engineered = _engineer_common_features(df)
    target_col = 'CalculatedPremiumPerTerm'
    if target_col not in engineered.columns:
        raise KeyError(f'Missing target column: {target_col}')

    df_premium = engineered.dropna(subset=[target_col]).copy()
    X = _select_feature_columns(df_premium, exclude=(target_col,))
    y = df_premium[target_col].astype(float)
    return X, y


def preprocess_for_claim_probability(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features/target for claim probability classification."""

    engineered = _engineer_common_features(df)
    X = _select_feature_columns(engineered)
    y = engineered['HasClaim'].astype(int)
    return X, y


def train_test_split_df(X, y, test_size=0.2, random_state=42, stratify=None):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def build_severity_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing pipeline for severity models."""

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def evaluate_regression_models(models: Dict[str, object], X_train, y_train, X_test, y_test, preprocessor):
    """Train and evaluate a dictionary of regression models."""

    results = []
    trained_models = {}
    for name, estimator in models.items():
        if estimator is None:
            continue

        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('model', estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results.append({'model': name, 'rmse': rmse, 'r2': r2})
        trained_models[name] = pipeline

    results_df = pd.DataFrame(results).sort_values('rmse') if results else pd.DataFrame()
    return results_df, trained_models


def build_default_severity_models(random_state: int = 42) -> Dict[str, object]:
    """Return baseline severity regressors including Linear, RF, and XGBoost."""

    models: Dict[str, object] = {
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if XGBRegressor is not None:
        models['XGBRegressor'] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            tree_method='hist',
            objective='reg:squarederror',
            reg_lambda=1.0,
        )
    else:
        models['XGBRegressor'] = None

    return models


def build_default_premium_models(random_state: int = 42) -> Dict[str, object]:
    """Return baseline premium regressors."""

    models: Dict[str, object] = {
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if XGBRegressor is not None:
        models['XGBRegressor'] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            tree_method='hist',
            objective='reg:squarederror',
            reg_lambda=1.0,
        )
    else:
        models['XGBRegressor'] = None

    return models


def build_default_classification_models(random_state: int = 42) -> Dict[str, object]:
    """Return baseline classifiers for claim probability."""

    models: Dict[str, object] = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'RandomForestClassifier': RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced',
        ),
    }

    if XGBClassifier is not None:
        models['XGBClassifier'] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            objective='binary:logistic',
            scale_pos_weight=1.0,
            tree_method='hist',
        )
    else:
        models['XGBClassifier'] = None

    return models


def evaluate_classification_models(
    models: Dict[str, object],
    X_train,
    y_train,
    X_test,
    y_test,
    preprocessor,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """Train classifiers and compute accuracy/precision/recall/F1."""

    results = []
    trained_models: Dict[str, Pipeline] = {}

    for name, estimator in models.items():
        if estimator is None:
            continue

        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('model', estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        results.append(
            {
                'model': name,
                'accuracy': accuracy_score(y_test, preds),
                'precision': precision_score(y_test, preds, zero_division=0),
                'recall': recall_score(y_test, preds, zero_division=0),
                'f1': f1_score(y_test, preds, zero_division=0),
            }
        )

        trained_models[name] = pipeline

    results_df = (
        pd.DataFrame(results).sort_values('f1', ascending=False)
        if results
        else pd.DataFrame()
    )
    return results_df, trained_models
