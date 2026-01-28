import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
import re

import yaml as yml

warnings.filterwarnings('ignore')


#local
from utils.preproc import load_data
from utils.train_eval import  train_and_evaluate, train_full_and_predict



# =============================================================================
# CONSTANTS
# =============================================================================

with open('config/config.yaml', 'r') as f:
    config = yml.safe_load(f)
    


LABEL_NAMES = config['LABEL_NAMES']

RSS_TO_LABEL = config['RSS_TO_LABEL']

RANDOM_STATE = config['RANDOM_STATE']

CONTEXT_INJECTION= config['CONTEXT_INJECTION']

USE_CATBOOST= config['USE_CATBOOST']

TUNING = config['TUNING']



# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("News Classification Pipeline")
    print("=" * 70)
    
    
    
    # Load data
    dev_df, eval_df = load_data(
        'data/development.csv',
        'data/evaluation.csv'
    )
    
    
    # Train and evaluate
    model, pipeline, metrics = train_and_evaluate(
        dev_df,
        test_size=0.2,
        model_type='svc',
        tune=TUNING,  
        context_injection=CONTEXT_INJECTION,  
        use_catboost=USE_CATBOOST,  
        save_dir='./models'
    )
    
    """
    # Generate submission
    submission = train_full_and_predict(
        dev_df,
        eval_df,
        model_type='svc',
        model = model,
        pipeline= pipeline,
        context_injection=CONTEXT_INJECTION,
        use_catboost=USE_CATBOOST,
        submission_path='submission.csv'
    )
    """
    