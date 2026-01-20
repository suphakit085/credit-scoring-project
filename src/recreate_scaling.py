import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

def create_features(df):
    """Recreate the feature engineering steps specific to the selected features."""
    df_new = df.copy()
    
    # Ratios
    # Safety +1 to avoid div by zero, matching notebook logic
    df_new['CREDIT_TO_ANNUITY_RATIO'] = df_new['AMT_CREDIT'] / (df_new['AMT_ANNUITY'] + 1)
    df_new['CREDIT_TO_GOODS_RATIO'] = df_new['AMT_CREDIT'] / (df_new['AMT_GOODS_PRICE'] + 1)
    
    # Time features
    df_new['AGE_YEARS'] = -df_new['DAYS_BIRTH'] / 365
    df_new['EMPLOYMENT_YEARS'] = -df_new['DAYS_EMPLOYED'] / 365
    df_new['EMPLOYMENT_TO_AGE_RATIO'] = df_new['EMPLOYMENT_YEARS'] / (df_new['AGE_YEARS'] + 1)
    
    # External Sources Stats
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df_new['EXT_SOURCE_MEAN'] = df_new[ext_cols].mean(axis=1)
    df_new['EXT_SOURCE_STD'] = df_new[ext_cols].std(axis=1)
    df_new['EXT_SOURCE_MIN'] = df_new[ext_cols].min(axis=1)
    df_new['EXT_SOURCE_MAX'] = df_new[ext_cols].max(axis=1)
    
    return df_new

def main():
    print("Loading data...")
    try:
        train_df = pd.read_csv('data/processed/train_cleaned.csv')
        feature_names_df = pd.read_csv('data/features/feature_names.csv')
        expected_features = feature_names_df['feature'].tolist()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Generating features...")
    # 1. Generate Domain Features
    df_fe = create_features(train_df)
    
    # 2. Categorical Encoding (Get Dummies)
    # We need to ensure we have columns like 'CODE_GENDER_M', 'NAME_FAMILY_STATUS_Married'
    # The clean way is to get_dummies for categorical columns
    print("Encoding categorical variables...")
    cat_cols = df_fe.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df_fe, columns=cat_cols, drop_first=True)
    
    # 3. Align with Expected Features
    # Create empty dataframe with expected columns
    df_final = pd.DataFrame(columns=expected_features)
    
    # Fill shared columns
    common_cols = list(set(df_encoded.columns) & set(expected_features))
    df_final[common_cols] = df_encoded[common_cols]
    
    # Fill missing columns with 0 (standard for OHE and safe for others validation)
    df_final = df_final.fillna(0)
    
    # Ensure correct order
    df_final = df_final[expected_features]
    
    print(f"Data aligned. Shape: {df_final.shape}")
    
    # 4. Impute & Scale
    print("Fitting Imputer and Scaler...")
    
    # Fit Imputer (Median) - RobustScaler doesn't like NaNs
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(df_final)
    
    # Fit Scaler (Robust)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # 5. Save Artifacts
    print("Saving artifacts...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(imputer, 'models/imputer.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    print("Success! 'imputer.joblib' and 'scaler.joblib' saved to models/.")

if __name__ == "__main__":
    main()
