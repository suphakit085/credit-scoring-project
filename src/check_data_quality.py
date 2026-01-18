import pandas as pd
import numpy as np

def check_data():
    print("Loading train_processed.csv...")
    try:
        df = pd.read_csv('data/processed/train_processed.csv')
    except FileNotFoundError:
        print("Error: data/processed/train_processed.csv not found.")
        return

    print(f"Data Shape: {df.shape}")
    
    # 1. Check DAYS_EMPLOYED anomaly
    if 'DAYS_EMPLOYED' in df.columns:
        max_days = df['DAYS_EMPLOYED'].max()
        print(f"\n1. DAYS_EMPLOYED Check:")
        print(f"   Max Value: {max_days}")
        if max_days > 365000:
            print("   [FAIL] Anomaly 365243 still present!")
        elif np.isnan(max_days):
             print("   [INFO] DAYS_EMPLOYED contains NaNs (Expected for retirees if replaced with NaN)")
        else:
            print("   [PASS] Anomaly 365243 seems removed (Max value reasonable or NaN).")
    
    # 2. Check Income Outliers
    if 'AMT_INCOME_TOTAL' in df.columns:
        max_inc = df['AMT_INCOME_TOTAL'].max()
        print(f"\n2. Income Check:")
        print(f"   Max Income: {max_inc:,.2f}")
        if max_inc > 10000000:
            print("   [WARN] Extremely high income detected. Consider capping.")
            
    # 3. Missing Values
    print(f"\n3. Missing Value Summary:")
    missing = df.isnull().mean()
    high_missing = missing[missing > 0.5]
    print(f"   Columns with > 50% missing: {len(high_missing)}")
    if len(high_missing) > 0:
        print(f"   Examples: {high_missing.index.tolist()[:5]}")
        
    # 4. Target Imbalance
    if 'TARGET' in df.columns:
        print(f"\n4. Target Distribution:")
        print(df['TARGET'].value_counts(normalize=True))

if __name__ == "__main__":
    check_data()
