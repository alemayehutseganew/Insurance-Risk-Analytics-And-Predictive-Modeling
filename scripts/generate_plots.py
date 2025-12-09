import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots():
    # Ensure output directory exists
    output_dir = os.path.join('reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    # Load Data
    print("Loading data...")
    try:
        df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|', low_memory=False)
    except FileNotFoundError:
        print("Data file not found.")
        return

    # Preprocessing for plots
    df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce')
    df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce')
    
    # 1. Univariate Analysis: Distribution of TotalPremium
    print("Generating Univariate Plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['TotalPremium'], bins=50, kde=True, log_scale=(False, True)) # Log scale for y to see tail
    plt.title('Distribution of Total Premium (Log Scale Y)')
    plt.xlabel('Total Premium')
    plt.ylabel('Count (Log)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'univariate_premium.png'))
    plt.close()

    # 2. Outlier Detection: Boxplot of TotalClaims
    print("Generating Outlier Plot...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['TotalClaims'])
    plt.title('Boxplot of Total Claims (Outlier Detection)')
    plt.xlabel('Total Claims')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outliers_claims.png'))
    plt.close()

    # 3. Geographic Trend: Claims by Province
    print("Generating Geographic Plot...")
    plt.figure(figsize=(12, 6))
    province_claims = df.groupby('Province')['TotalClaims'].mean().sort_values(ascending=False)
    sns.barplot(x=province_claims.index, y=province_claims.values)
    plt.title('Average Total Claims by Province')
    plt.xlabel('Province')
    plt.ylabel('Average Total Claims')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'geo_province_claims.png'))
    plt.close()

    # 4. Bivariate Analysis: Premium vs Claims Scatter (Sampled)
    print("Generating Bivariate Plot...")
    plt.figure(figsize=(10, 6))
    sample_df = df.sample(n=10000, random_state=42) # Sample for performance and clarity
    sns.scatterplot(x='TotalPremium', y='TotalClaims', data=sample_df, alpha=0.5)
    plt.title('Total Premium vs Total Claims (Sampled n=10,000)')
    plt.xlabel('Total Premium')
    plt.ylabel('Total Claims')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bivariate_premium_claims.png'))
    plt.close()

    print("All plots generated successfully.")

if __name__ == "__main__":
    generate_plots()
