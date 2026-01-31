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
from utils.train_eval import  train_and_evaluate, train_full_and_predict, compare_models



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

USE_CHI2 = config['USE_CHI2']

CHI2_K = config['CHI2_PARAMS']['chi2_k']

TUNING = config['TUNING']

TFIDF_PARAMS = config['TFIDF_PARMAS']



# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("News Classification Pipeline")
    print("=" * 70)
    
    TFIDF_PARAMS['ngram_range'] = eval(TFIDF_PARAMS['ngram_range'])
    TFIDF_PARAMS['token_pattern'] = eval(TFIDF_PARAMS['token_pattern'])
    
    
    
    
    
    # Load data
    dev_df, eval_df = load_data(
        'data/development.csv',
        'data/evaluation.csv'
    )
    
    """
    experiments = {
    #'base_tfidf': {'context_injection': False, 'use_catboost': False, 'use_chi': False},
    #'context_inj_only': {'context_injection': True, 'use_catboost': False, 'use_chi': False},
    'cateboost_only': {'context_injection': False, 'use_catboost': True, 'use_chi': False},
    #'chi2_only': {'context_injection': False, 'use_catboost': False, 'use_chi': True},
    'inj_n_catboost': {'context_injection': True, 'use_catboost': True, 'use_chi': False},
    #'inj_n_chi': {'context_injection': True, 'use_catboost': False, 'use_chi': True},
    #'cat_n_chi': {'context_injection': False, 'use_catboost': True, 'use_chi': True},
    'full_pipeline': {'context_injection': True, 'use_catboost': True, 'use_chi': True},
}

    
    results = {}
    for name, dict_exp in experiments.items():
        model, pipeline, metrics = train_and_evaluate(
        dev_df,
        test_size=0.2,
        model_type='svc', #lgbm, nb, logit
        tune=False,  
        tfidf_params=TFIDF_PARAMS,
        context_injection=dict_exp['context_injection'],  
        use_catboost=dict_exp['use_catboost'],  
        use_chi2=dict_exp['use_chi'],
        chi2_k=CHI2_K,
        save_dir='./models'
    )
        results[name] = metrics['macro_f1']
        print(f"{name}: {metrics['macro_f1']:.4f}")
    # Train and evaluate
    
    comparison = compare_models(
        dev_df,
        model_types=['svc', 'logit', 'lgbm'],
        tune=False,
        tfidf_params=TFIDF_PARAMS,
        use_catboost=False,
        context_injection=True,
        use_chi2=False,
        chi2_k=CHI2_K
    ) 
    """
    
    model, pipeline, metrics = train_and_evaluate(
        dev_df,
        test_size=0.2,
        model_type='svc', #svc, nb, logit, lgbm
        tune=False,  
        tfidf_params=TFIDF_PARAMS, #15,000 (1,3)
        context_injection=CONTEXT_INJECTION,  #True
        use_catboost=USE_CATBOOST,  #True
        use_chi2=USE_CHI2, #False
        chi2_k=CHI2_K, #10,000
        save_dir='./models'
    )
    # Generate submission
    submission = train_full_and_predict(
        dev_df,
        eval_df,
        model_type='svc',
        model = model,
        pipeline= None,
        context_injection=CONTEXT_INJECTION,
        use_catboost=USE_CATBOOST,
        use_chi2=USE_CHI2,
        chi2_k=CHI2_K,
        submission_path='submission.csv'
    )
  
    
    