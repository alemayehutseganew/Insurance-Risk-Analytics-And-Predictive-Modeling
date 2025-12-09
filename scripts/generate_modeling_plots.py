import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.modeling import load_data, preprocess_for_premium, build_severity_preprocessor

# Setup directories
FIGURES_DIR = Path(__file__).resolve().parents[1] / 'reports' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def generate_plots():
    print("Loading data...")
    # Load a sample for speed
    df = load_data(nrows=50000)
    
    print("Preprocessing...")
    X, y = preprocess_for_premium(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessor
    preprocessor = build_severity_preprocessor(X_train)
    
    # Fit preprocessor
    print("Fitting preprocessor...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f"feat_{i}" for i in range(X_train_transformed.shape[1])]
        
    # Train Model
    print("Training Random Forest...")
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_transformed, y_train)
    
    # 1. Feature Importance (Permutation)
    print("Generating Feature Importance Plot...")
    result = permutation_importance(model, X_test_transformed, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()[-10:] # Top 10
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title("Top 10 Features (Permutation Importance)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_importance.png')
    plt.close()
    
    # 2. SHAP
    print("Generating SHAP Summary Plot...")
    # Use a small background sample for speed
    background = X_train_transformed[:100]
    # Convert sparse matrix to dense if necessary
    if hasattr(background, "toarray"):
        background = background.toarray()
        
    X_test_sample = X_test_transformed[:200]
    if hasattr(X_test_sample, "toarray"):
        X_test_sample = X_test_sample.toarray()

    explainer = shap.Explainer(model, background)
    shap_values = explainer(X_test_sample) # Explain 200 instances
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'shap_summary.png')
    plt.close()
    
    # 3. PDP
    print("Generating PDP Plot...")
    # Find index of 'kilowatts' or similar numeric feature
    target_feature = None
    for i, name in enumerate(feature_names):
        if 'kilowatts' in name.lower() or 'vehicleage' in name.lower():
            target_feature = i
            break
            
    if target_feature is not None:
        plt.figure(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(
            model, 
            X_test_transformed, 
            [target_feature], 
            feature_names=feature_names,
            kind='average'
        )
        plt.title(f"Partial Dependence Plot: {feature_names[target_feature]}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'pdp_plot.png')
        plt.close()
    
    # 4. LIME
    print("Generating LIME Explanation...")
    # LIME requires dense arrays
    training_data_dense = X_train_transformed
    if hasattr(training_data_dense, "toarray"):
        training_data_dense = training_data_dense.toarray()
        
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(training_data_dense),
        feature_names=feature_names,
        mode='regression'
    )
    
    # Explain one instance
    instance_dense = X_test_transformed[0]
    if hasattr(instance_dense, "toarray"):
        instance_dense = instance_dense.toarray()
        
    exp = explainer_lime.explain_instance(
        data_row=instance_dense.flatten(), # Ensure 1D array
        predict_fn=model.predict
    )
    
    # Save as plot
    plt.figure(figsize=(10, 6))
    exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'lime_explanation.png')
    plt.close()
    
    print("All plots generated successfully.")

if __name__ == "__main__":
    generate_plots()
