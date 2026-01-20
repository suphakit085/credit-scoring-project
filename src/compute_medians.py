
import pandas as pd
import joblib
import json

def main():
    print("Loading training data...")
    try:
        df = pd.read_csv('data/processed/train_cleaned.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Computing medians...")
    # Compute median for all numeric columns
    medians = df.median(numeric_only=True).to_dict()
    
    # Save to JSON for App to load
    with open('data/processed/feature_medians.json', 'w') as f:
        json.dump(medians, f)
        
    print(f"Saved {len(medians)} medians to data/processed/feature_medians.json")
    
    # Print top 20 medians for verification
    print("\nSample Medians (Top features):")
    top_feats = ['CREDIT_TO_GOODS_RATIO', 'EXT_SOURCE_MEAN', 'PREV_CNT_PAYMENT_max', 
                 'PREV_DAYS_FIRST_DRAWING_sum', 'DEF_30_CNT_SOCIAL_CIRCLE']
    
    for feat in top_feats:
        if feat in medians:
            print(f"  {feat}: {medians[feat]}")
        else:
            print(f"  {feat}: Not found (maybe derived later?)")

if __name__ == "__main__":
    main()
