
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import joblib
import json
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
import re

import yaml as yml

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================

with open('config/config.yaml', 'r') as f:
    config = yml.safe_load(f)
    




RANDOM_STATE = config['RANDOM_STATE']

np.random.seed(RANDOM_STATE)


def load_data(
    dev_path: str | Path,
    eval_path: str | Path | None = None,
    **kwargs
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load development and evaluation datasets."""
    dev_df = pd.read_csv(dev_path, **kwargs)
    print(f"Development set: {dev_df.shape[0]:,} samples, {dev_df.shape[1]} columns")
    
    eval_df = None
    if eval_path is not None:
        eval_df = pd.read_csv(eval_path, **kwargs)
        print(f"Evaluation set: {eval_df.shape[0]:,} samples, {eval_df.shape[1]} columns")
    
    return dev_df, eval_df


def preprocess_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Preprocess DataFrame by handling duplicates and cleaning.
    
    For training data:
    - Removes exact duplicates (keeping most recent)
    - Removes ambiguous samples (same text, different labels)
    """
    temp_df = df.copy()
    original_len = len(temp_df)
    
    # Convert timestamp
    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], errors='coerce')
    
    if is_train and 'label' in temp_df.columns:
        # Sort by timestamp (most recent first)
        temp_df = temp_df.sort_values('timestamp', ascending=False, na_position='last')
        
        # Remove exact duplicates
        temp_df = temp_df.drop_duplicates(
            subset=['source', 'title', 'article', 'label'], 
            keep='first'
        )
        removed = original_len - len(temp_df)
        print(f"Removed {removed:,} exact duplicates")
        
        # Remove duplicates with same title/article but different sources
        original_len = len(temp_df)
        temp_df = temp_df.drop_duplicates(
            subset=['title', 'article', 'label'], 
            keep='first'
        )
        removed = original_len - len(temp_df)
        print(f"Removed {removed:,} title/article duplicates")
        
        # Remove ambiguous samples (same content, different labels)
        original_len = len(temp_df)
        label_counts = temp_df.groupby(['source', 'title', 'article'])['label'].transform('nunique')
        temp_df = temp_df[label_counts == 1]
        removed = original_len - len(temp_df)
        print(f"Removed {removed:,} ambiguous samples")
        
        # Shuffle
        temp_df = temp_df.sample(frac=1, random_state=RANDOM_STATE)
    
    temp_df = temp_df.reset_index(drop=True)
    print(f"Final shape: {temp_df.shape[0]:,} samples")
    
    return temp_df