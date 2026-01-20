import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path so we can import recreate_scaling
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from recreate_scaling import create_features

def test_create_features_ratios():
    """Test ratio feature creation including handling of zero division."""
    data = {
        'AMT_CREDIT': [10000.0, 20000.0],
        'AMT_ANNUITY': [1000.0, 0.0],  # 0.0 to test division by zero protection (+1)
        'AMT_GOODS_PRICE': [5000.0, 0.0],
        'DAYS_BIRTH': [-10000, -20000],
        'DAYS_EMPLOYED': [-1000, -2000],
        'EXT_SOURCE_1': [0.1, 0.2],
        'EXT_SOURCE_2': [0.1, 0.2],
        'EXT_SOURCE_3': [0.1, 0.2]
    }
    df = pd.DataFrame(data)
    
    df_new = create_features(df)
    
    # Check CREDIT_TO_ANNUITY_RATIO
    expected_ratio_1 = 10000.0 / (1000.0 + 1)
    assert np.isclose(df_new['CREDIT_TO_ANNUITY_RATIO'][0], expected_ratio_1)
    
    # Check CREDIT_TO_GOODS_RATIO
    expected_ratio_2 = 10000.0 / (5000.0 + 1)
    assert np.isclose(df_new['CREDIT_TO_GOODS_RATIO'][0], expected_ratio_2)
    
    # Check division by zero handling (denominator + 1)
    # If annuity is 0, denominator is 1. Ratio = Credit / 1
    assert df_new['CREDIT_TO_ANNUITY_RATIO'][1] == 20000.0
    assert df_new['CREDIT_TO_GOODS_RATIO'][1] == 20000.0

def test_create_features_time():
    """Test time-based features."""
    data = {
        'AMT_CREDIT': [10000.0],
        'AMT_ANNUITY': [1000.0],
        'AMT_GOODS_PRICE': [5000.0],
        'DAYS_BIRTH': [-3650],  # 10 years
        'DAYS_EMPLOYED': [-365], # 1 year
        'EXT_SOURCE_1': [0.5],
        'EXT_SOURCE_2': [0.5],
        'EXT_SOURCE_3': [0.5]
    }
    df = pd.DataFrame(data)
    df_new = create_features(df)
    
    assert df_new['AGE_YEARS'][0] == 10.0
    assert df_new['EMPLOYMENT_YEARS'][0] == 1.0
    
    # EMPLOYMENT_TO_AGE_RATIO = 1.0 / (10.0 + 1) = 1/11
    assert np.isclose(df_new['EMPLOYMENT_TO_AGE_RATIO'][0], 1.0/11.0)

def test_create_features_ext_sources():
    """Test external source statistics."""
    data = {
        'AMT_CREDIT': [10000.0],
        'AMT_ANNUITY': [1000.0],
        'AMT_GOODS_PRICE': [5000.0],
        'DAYS_BIRTH': [-10000],
        'DAYS_EMPLOYED': [-1000],
        'EXT_SOURCE_1': [0.2],
        'EXT_SOURCE_2': [0.5],
        'EXT_SOURCE_3': [0.8]
    }
    df = pd.DataFrame(data)
    df_new = create_features(df)
    
    ext_vals = [0.2, 0.5, 0.8]
    
    assert np.isclose(df_new['EXT_SOURCE_MEAN'][0], np.mean(ext_vals))
    assert np.isclose(df_new['EXT_SOURCE_STD'][0], np.std(ext_vals, ddof=1)) # Pandas std is sample std by default
    assert df_new['EXT_SOURCE_MIN'][0] == 0.2
    assert df_new['EXT_SOURCE_MAX'][0] == 0.8

def test_create_features_preserves_rows():
    """Ensure row count check."""
    data = pd.DataFrame({
        'AMT_CREDIT': range(10),
        'AMT_ANNUITY': range(10),
        'AMT_GOODS_PRICE': range(10),
        'DAYS_BIRTH': range(10),
        'DAYS_EMPLOYED': range(10),
        'EXT_SOURCE_1': range(10),
        'EXT_SOURCE_2': range(10),
        'EXT_SOURCE_3': range(10)
    })
    df_new = create_features(data)
    assert len(df_new) == 10
