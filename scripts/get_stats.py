import pandas as pd
try:
    df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|', low_memory=False)
    
    # Basic Stats
    rows = len(df)
    cols = len(df.columns)
    
    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    
    # Describe
    desc = df[['TotalPremium', 'TotalClaims']].describe()
    
    with open('stats.txt', 'w') as f:
        f.write(f"Rows: {rows}\n")
        f.write(f"Cols: {cols}\n")
        f.write("Missing Values:\n")
        f.write(missing.to_string())
        f.write("\n\nDescriptive Stats:\n")
        f.write(desc.to_string())
        
    print("Stats written to stats.txt")
except Exception as e:
    print(e)
