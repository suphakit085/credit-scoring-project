
import joblib
import pandas as pd
import numpy as np

def main():
    print("Loading model...")
    try:
        model = joblib.load('models/best_model_lgbm.pkl')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading feature names...")
    try:
        feature_names_df = pd.read_csv('data/features/feature_names.csv')
        feature_names = feature_names_df['feature'].tolist()
    except Exception as e:
        print(f"Error loading feature names: {e}")
        return

    print(f"Model type: {type(model)}")
    
    # Check if model has feature_importances_
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create DataFrame
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        fi_df = fi_df.sort_values(by='importance', ascending=False)
        
        print("\nTop 20 Features by Importance:")
        print(fi_df.head(20))
        
        # Also print bottom 10 just in case
        # print("\nBottom 10 Features:")
        # print(fi_df.tail(10))
    else:
        print("Model does not have feature_importances_ attribute.")

if __name__ == "__main__":
    main()
