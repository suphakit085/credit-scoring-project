import pandas as pd
import numpy as np
import os

# Identify categorical columns (object type)
def get_categorical_columns(df):
    return df.select_dtypes(include=['object']).columns.tolist()

def impute_missing_values(df):
    """
    Impute missing values in the dataframe.
    - Numeric columns: Median
    - Categorical columns: Mode (most frequent) or a specific placeholder string
    """
    print("Imputing missing values...")
    df_imputed = df.copy()
    
    # numeric imputation
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_imputed[col].isnull().any():
            median_val = df_imputed[col].median()
            df_imputed[col].fillna(median_val, inplace=True)
            
    # categorical imputation
    cat_cols = df_imputed.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df_imputed[col].isnull().any():
            # Use mode, or a placeholder like 'Unknown'
            # For simplicity, using mode
            if not df_imputed[col].mode().empty:
                mode_val = df_imputed[col].mode()[0]
                df_imputed[col].fillna(mode_val, inplace=True)
            else:
                df_imputed[col].fillna("Unknown", inplace=True)
                
    return df_imputed

def encode_features(df):
    """
    Encode categorical features.
    - 2 categories: Label Encoding (0/1)
    - >2 categories: One-Hot Encoding
    """
    print("Encoding categorical features...")
    df_encoded = df.copy()
    le_count = 0
    
    cat_cols = get_categorical_columns(df_encoded)
    
    for col in cat_cols:
        # Label Encode if 2 or fewer unique values
        if len(df_encoded[col].unique()) <= 2:
            # Simple factorization
            df_encoded[col], _ = pd.factorize(df_encoded[col])
            le_count += 1
    
    # One-Hot Encode the rest
    df_encoded = pd.get_dummies(df_encoded)
    
    print(f"{le_count} columns were label encoded.")
    print(f"Total columns after one-hot encoding: {df_encoded.shape[1]}")
    
    return df_encoded

def load_data(data_dir):
    """
    Load all datasets from the specified directory.
    
    Args:
        data_dir (str): Path to the directory containing raw CSV files.
        
    Returns:
        dict: A dictionary containing pandas DataFrames for each dataset.
    """
    data = {}
    files = [
        "application_train.csv",
        "application_test.csv",
        "bureau.csv",
        "bureau_balance.csv",
        "credit_card_balance.csv",
        "installments_payments.csv",
        "previous_application.csv",
        "POS_CASH_balance.csv"
    ]
    
    print(f"Loading data from {data_dir}...")
    for file in files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            key = file.replace(".csv", "")
            print(f"Loading {file}...")
            data[key] = pd.read_csv(file_path)
        else:
            print(f"Warning: {file} not found in {data_dir}")
            
    return data

def clean_application_data(df):
    """
    Perform initial data cleaning on the application dataset.
    
    Args:
        df (pd.DataFrame): The application_train or application_test dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df_clean = df.copy()
    
    # 1. Handle DAYS_EMPLOYED anomaly
    # 365243 days is often used as a placeholder for null/retired in this dataset
    df_clean['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    
    # 2. Flag for potential anomalies (optional but good practice)
    df_clean['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243)
    
    # 3. Impute Missing Values
    df_clean = impute_missing_values(df_clean)

    # 4. Categorical encoding
    df_clean = encode_features(df_clean)
    
    return df_clean

def preprocess_bureau_data(bureau):
    """
    Preprocess and aggregate bureau data.
    Args:
        bureau (pd.DataFrame): Bureau data.
    Returns:
        pd.DataFrame: Aggregated bureau data with SK_ID_CURR as index.
    """
    print("Preprocessing bureau data...")
    # One-hot encoding for categorical columns (optional, keeping it simple for now)
    
    # Select numeric features for aggregation
    numeric_cols = bureau.select_dtypes(include=[np.number]).columns.tolist()
    # Ensure SK_ID_CURR is kept if it's numeric (it is)
    
    # Aggregating numeric features
    # SK_ID_BUREAU is ID, dropping it. SK_ID_CURR is grouping key.
    cols_to_agg = [c for c in numeric_cols if c != 'SK_ID_BUREAU' and c != 'SK_ID_CURR']
    
    bureau_agg = bureau.groupby('SK_ID_CURR')[cols_to_agg].agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
    
    # Rename columns
    columns = ['SK_ID_CURR']
    for col in bureau_agg.columns.levels[0]:
        if col != 'SK_ID_CURR':
            for stat in bureau_agg.columns.levels[1][:-1]: # excluding the empty string from index
                columns.append('BUREAU_%s_%s' % (col, stat))
    
    bureau_agg.columns = columns
    return bureau_agg

def preprocess_previous_applications(prev_app):
    """
    Preprocess and aggregate previous application data.
    """
    print("Preprocessing previous applications...")
    
    # Select numeric columns
    numeric_cols = prev_app.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_agg = [c for c in numeric_cols if c != 'SK_ID_PREV' and c != 'SK_ID_CURR']

    # Simple numeric aggregation
    prev_agg = prev_app.groupby('SK_ID_CURR')[cols_to_agg].agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
    
    # Rename columns (flatten MultiIndex)
    prev_agg.columns = ['SK_ID_CURR'] + ['PREV_' + '_'.join(col).strip('_') for col in prev_agg.columns.values if col[0] != 'SK_ID_CURR']
    
    return prev_agg

def merge_data(app_data, bureau_agg, prev_agg):
    """
    Merge aggregated features into the main application dataset.
    """
    print("Merging datasets...")
    df = app_data.merge(bureau_agg, on='SK_ID_CURR', how='left')
    df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
    return df

def main():
    # Paths relative to the project root (where the script is executed)
    DATA_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    # 1. Load Data
    # Only loading necessary files to save memory if needed, but load_data loads all.
    # We'll use the existing load_data.
    data = load_data(DATA_DIR)
    
    # 2. Clean Application Data
    print("Cleaning application data...")
    if 'application_train' in data:
        df_train = clean_application_data(data['application_train'])
    if 'application_test' in data:
        df_test = clean_application_data(data['application_test'])
    
    # 3. Preprocess Secondary Tables
    bureau_agg = None
    if 'bureau' in data:
        bureau_agg = preprocess_bureau_data(data['bureau'])
        
    prev_agg = None
    if 'previous_application' in data:
        prev_agg = preprocess_previous_applications(data['previous_application'])
        
    # 4. Merge
    print("Merging training data...")
    feature_dfs = [df for df in [bureau_agg, prev_agg] if df is not None]
    
    if 'application_train' in data:
        df_train_merged = df_train
        for feat_df in feature_dfs:
            df_train_merged = df_train_merged.merge(feat_df, on='SK_ID_CURR', how='left')
        
        print(f"Saving train data to {PROCESSED_DIR}...")
        df_train_merged.to_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"), index=False)
        print(f"Train shape: {df_train_merged.shape}")

    if 'application_test' in data:
        print("Merging test data...")
        df_test_merged = df_test
        for feat_df in feature_dfs:
            df_test_merged = df_test_merged.merge(feat_df, on='SK_ID_CURR', how='left')
            
        print(f"Saving test data to {PROCESSED_DIR}...")
        df_test_merged.to_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"), index=False)
        print(f"Test shape: {df_test_merged.shape}")
        
    print("Data processing complete.")

if __name__ == "__main__":
    main()
