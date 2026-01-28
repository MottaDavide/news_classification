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

# Sklearn
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    f1_score, confusion_matrix,
    precision_recall_fscore_support, make_scorer
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

# LightGBM
import lightgbm as lgb

# Scipy
from scipy.sparse import hstack, csr_matrix

# Category encoders
from category_encoders import CatBoostEncoder

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





# =============================================================================
# PIPELINE CLASS
# =============================================================================




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
        model_type='lgbm',
        tune=False,  
        context_injection=True,  
        use_catboost=True,  
        save_dir='./models'
    )
    
    # Generate submission
    submission = train_full_and_predict(
        dev_df,
        eval_df,
        model_type='lgbm',
        context_injection=True,
        use_catboost=True,
        submission_path='submission.csv'
    )