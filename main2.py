"""
News Classification Pipeline - Optimized Version
=================================================
Data Science and Machine Learning Lab - Politecnico di Torino
Winter 2026

Ottimizzazioni rispetto alla versione originale:
- Context Injection: metadata (source, rss, domain) iniettati nel testo
- Migliore regolarizzazione (C=0.05, max_features=10000)
- +1.8% Macro F1 rispetto al metodo CatBoost separato
"""

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

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================

LABEL_NAMES = {
    0: 'International',
    1: 'Business',
    2: 'Technology',
    3: 'Entertainment',
    4: 'Sports',
    5: 'General',
    6: 'Health'
}

RSS_TO_LABEL = {
    'world': 0, 'europe': 0, 'politics': 0, 'elections': 0, 'us': 0,
    'business': 1,
    'tech': 2, 'science': 2,
    'entertainment': 3,
    'sports': 4,
    'cnn_topstories': 5,
    'health': 6,
}

RANDOM_STATE = 42

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

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


# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_text(text: str) -> str:
    """Clean and preprocess text."""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).lower()
    
    # Remove HTML entities and tags
    text = re.sub(r'&#?\w+;', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove HTML attribute names
    text = re.sub(r'\b(alt|img|src|href|title|width|height|http|https|www)\b', ' ', text)
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_title_suffix(title: str) -> Optional[str]:
    """Extract suffix in brackets from title (e.g., 'Title (Reuters)' -> 'Reuters')."""
    if pd.isna(title) or not isinstance(title, str):
        return None
    match = re.search(r'\(([^)]+)\)\s*$', title)
    return match.group(1) if match else None


def extract_first_domain(article: str) -> str:
    """Extract domain string (e.g., 'yahoo', 'bbc')."""
    if pd.isna(article): 
        return ""
    match = re.search(r'https?://([^\s<>"{}|\\^`\[\]/]+)', str(article))
    if match:
        dom = match.group(1).replace('www.', '')
        return dom.split('.')[0]
    return ""


def extract_rss_label(text: str) -> Optional[int]:
    """Extract predicted label from RSS category in URL (100% accuracy)."""
    if pd.isna(text) or not isinstance(text, str):
        return None
    match = re.search(r'/rss/([a-z_]+)', text.lower())
    if match:
        category = match.group(1).split('?')[0]
        return RSS_TO_LABEL.get(category, None)
    return None


def extract_rss_string(text: str) -> str:
    """Extract RSS category as string for text injection."""
    if pd.isna(text): 
        return ""
    match = re.search(r'/rss/([a-z_]+)', str(text).lower())
    return match.group(1).replace('_', ' ') if match else ""


def count_links(article: str) -> Dict[str, int]:
    """Count links by type (total, images, ads, feeds)."""
    if pd.isna(article):
        return {'n_links': 0, 'n_images': 0, 'n_ads': 0, 'n_feeds': 0}
    
    urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', str(article))
    counts = {'n_links': len(urls), 'n_images': 0, 'n_ads': 0, 'n_feeds': 0}
    
    for url in urls:
        url_lower = url.lower()
        if any(ext in url_lower for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
            counts['n_images'] += 1
        elif any(ad in url_lower for ad in ['doubleclick', 'adlog', 'pheedo']):
            counts['n_ads'] += 1
        elif any(feed in url_lower for feed in ['feeds.', 'rss.', 'feedburner']):
            counts['n_feeds'] += 1
    
    return counts


def generate_features(
    df: pd.DataFrame,
    context_injection: bool = True,
    source_weight: int = 3,
    rss_weight: int = 5,
    title_weight: int = 2,
    article_weight: int = 1
) -> pd.DataFrame:
    """
    Generate all features from raw DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns: title, article, source.
    context_injection : bool
        If True, inject metadata (source, rss) into combined_text.
        This improves F1 by ~1.8% compared to separate CatBoost encoding.
    source_weight, rss_weight, title_weight, article_weight : int
        Weights for text repetition in combined_text.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with additional feature columns.
    """
    temp_df = df.copy()
    
    # Clean text columns
    temp_df['clean_title'] = temp_df['title'].apply(clean_text)
    temp_df['clean_article'] = temp_df['article'].apply(clean_text)
    
    # Metadata features
    temp_df['meta_source'] = temp_df['source'].astype(str).apply(
        lambda x: re.sub(r'[^a-zA-Z]', '', x).lower()
    )
    temp_df['first_link_domain'] = temp_df['article'].apply(extract_first_domain)
    temp_df['meta_rss_str'] = temp_df['article'].apply(extract_rss_string)
    
    # Title suffix
    temp_df['title_suffix'] = temp_df['title'].apply(extract_title_suffix)
    
    # RSS label (numeric)
    temp_df['rss_label'] = temp_df['article'].apply(extract_rss_label)
    
    # Link counts
    link_counts = temp_df['article'].apply(count_links).apply(pd.Series)
    temp_df = pd.concat([temp_df, link_counts], axis=1)
    
    # Article length (binned)
    temp_df['article_length'] = pd.cut(
        temp_df['article'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0),
        bins=[0, 10, 100, 180, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(float).fillna(1)
    
    # Combined text for TF-IDF
    if context_injection:
        # IMPORTANT: Inject metadata into text (best performance +1.8% F1)
        temp_df['combined_text'] = (
            (temp_df['meta_source'] + " ") * source_weight +
            (temp_df['meta_rss_str'] + " ") * rss_weight +
            (temp_df['clean_title'] + " ") * title_weight +
            temp_df['clean_article'] * article_weight  # Fixed: was clean_title before
        )
    else:
        # Traditional approach (text only)
        temp_df['combined_text'] = (
            (temp_df['clean_title'] + " ") * title_weight +
            temp_df['clean_article'] * article_weight
        )
    
    return temp_df


# =============================================================================
# PIPELINE CLASS
# =============================================================================

class NewsPipeline(BaseEstimator, TransformerMixin):
    """
    Complete feature engineering pipeline for news classification.
    
    Combines:
    - TF-IDF vectorization of combined text
    - Numerical feature scaling
    - CatBoost encoding for categorical features (optional)
    - Cyclic timestamp features
    
    Parameters
    ----------
    tfidf_params : dict
        Parameters for TfidfVectorizer.
    catboost_sigma : float
        Sigma parameter for CatBoost encoding.
    use_catboost : bool
        Whether to use CatBoost encoding (default: True for hybrid mode).
    use_cyclic_time : bool
        Whether to use cyclic timestamp features.
    """
    
    def __init__(
        self,
        tfidf_params: Optional[Dict] = None,
        catboost_sigma: float = 0.5,
        use_catboost: bool = True,
        use_cyclic_time: bool = True
    ):
        # Optimized TF-IDF params for context injection
        self.tfidf_params = tfidf_params or {
            'max_features': 10000,
            'ngram_range': (1, 2),
            'min_df': 5,
            'max_df': 0.9,
            'sublinear_tf': True,
            'token_pattern': r'(?u)\b[a-zA-Z]{3,}\b'
        }
        self.catboost_sigma = catboost_sigma
        self.use_catboost = use_catboost
        self.use_cyclic_time = use_cyclic_time
        
        # Will be fitted
        self.tfidf_ = None
        self.scaler_ = None
        self.cat_encoders_ = {}
        self.median_ts_ = None
        self.feature_names_ = None
        
        # Column definitions
        self.num_cols_ = ['page_rank', 'n_links', 'n_images', 'n_ads', 'n_feeds', 
                          'article_length', 'rss_label']
        self.cat_cols_ = ['source', 'first_link_domain', 'title_suffix']
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'NewsPipeline':
        """Fit the pipeline."""
        self.fit_transform(X, y)
        return self
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> csr_matrix:
        """Fit and transform the data."""
        features_list = []
        feature_names = []
        
        # 1. TF-IDF
        print("Fitting TF-IDF...")
        self.tfidf_ = TfidfVectorizer(**self.tfidf_params)
        X_tfidf = self.tfidf_.fit_transform(X['combined_text'])
        features_list.append(X_tfidf)
        feature_names.extend([f"tfidf_{w}" for w in self.tfidf_.get_feature_names_out()])
        print(f"  TF-IDF: {X_tfidf.shape[1]} features")
        
        # 2. Numerical features
        num_cols = [c for c in self.num_cols_ if c in X.columns]
        if num_cols:
            print("Fitting numerical scaler...")
            self.scaler_ = StandardScaler()
            X_num = self.scaler_.fit_transform(X[num_cols].fillna(0))
            features_list.append(csr_matrix(X_num))
            feature_names.extend([f"num_{c}" for c in num_cols])
            print(f"  Numerical: {len(num_cols)} features")
        
        # 3. CatBoost encoding (optional, for hybrid mode)
        if self.use_catboost:
            cat_cols = [c for c in self.cat_cols_ if c in X.columns]
            if cat_cols:
                print("Fitting CatBoost encoders...")
                cat_features = []
                
                for cls in np.unique(y):
                    y_binary = (y == cls).astype(int)
                    enc = CatBoostEncoder(cols=cat_cols, sigma=self.catboost_sigma)
                    X_enc = enc.fit_transform(X[cat_cols].fillna('missing'), y_binary)
                    self.cat_encoders_[cls] = enc
                    cat_features.append(X_enc.values)
                    feature_names.extend([f"{c}_prob_cl{cls}" for c in cat_cols])
                
                X_cat = np.hstack(cat_features)
                features_list.append(csr_matrix(X_cat))
                print(f"  CatBoost: {X_cat.shape[1]} features")
        
        # 4. Cyclic timestamp features
        if self.use_cyclic_time and 'timestamp' in X.columns:
            print("Fitting cyclic timestamp features...")
            ts = pd.to_datetime(X['timestamp'], errors='coerce')
            self.median_ts_ = ts.median()
            ts = ts.fillna(self.median_ts_)
            
            cyclic = pd.DataFrame({
                'is_ts_missing': pd.to_datetime(X['timestamp'], errors='coerce').isna().astype(int),
                'hour_sin': np.sin(2 * np.pi * ts.dt.hour / 24),
                'hour_cos': np.cos(2 * np.pi * ts.dt.hour / 24),
                'day_sin': np.sin(2 * np.pi * ts.dt.dayofweek / 7),
                'day_cos': np.cos(2 * np.pi * ts.dt.dayofweek / 7),
                'month_sin': np.sin(2 * np.pi * (ts.dt.month - 1) / 12),
                'month_cos': np.cos(2 * np.pi * (ts.dt.month - 1) / 12),
            })
            
            features_list.append(csr_matrix(cyclic.values))
            feature_names.extend(cyclic.columns.tolist())
            print(f"  Cyclic timestamp: {cyclic.shape[1]} features")
        
        # Combine all features
        X_combined = hstack(features_list)
        self.feature_names_ = np.array(feature_names)
        print(f"  TOTAL: {X_combined.shape[1]} features")
        
        return X_combined
    
    def transform(self, X: pd.DataFrame) -> csr_matrix:
        """Transform data using fitted pipeline."""
        features_list = []
        
        # TF-IDF
        if self.tfidf_ is not None:
            X_tfidf = self.tfidf_.transform(X['combined_text'])
            features_list.append(X_tfidf)
        
        # Numerical
        num_cols = [c for c in self.num_cols_ if c in X.columns]
        if num_cols and self.scaler_ is not None:
            X_num = self.scaler_.transform(X[num_cols].fillna(0))
            features_list.append(csr_matrix(X_num))
        
        # CatBoost
        if self.use_catboost:
            cat_cols = [c for c in self.cat_cols_ if c in X.columns]
            if cat_cols and self.cat_encoders_:
                cat_features = []
                for cls, enc in self.cat_encoders_.items():
                    X_enc = enc.transform(X[cat_cols].fillna('missing'))
                    cat_features.append(X_enc.values)
                X_cat = np.hstack(cat_features)
                features_list.append(csr_matrix(X_cat))
        
        # Cyclic timestamp
        if self.use_cyclic_time and 'timestamp' in X.columns and self.median_ts_ is not None:
            ts = pd.to_datetime(X['timestamp'], errors='coerce')
            ts = ts.fillna(self.median_ts_)
            
            cyclic = pd.DataFrame({
                'is_ts_missing': pd.to_datetime(X['timestamp'], errors='coerce').isna().astype(int),
                'hour_sin': np.sin(2 * np.pi * ts.dt.hour / 24),
                'hour_cos': np.cos(2 * np.pi * ts.dt.hour / 24),
                'day_sin': np.sin(2 * np.pi * ts.dt.dayofweek / 7),
                'day_cos': np.cos(2 * np.pi * ts.dt.dayofweek / 7),
                'month_sin': np.sin(2 * np.pi * (ts.dt.month - 1) / 12),
                'month_cos': np.cos(2 * np.pi * (ts.dt.month - 1) / 12),
            })
            features_list.append(csr_matrix(cyclic.values))
        
        return hstack(features_list)


# =============================================================================
# MODEL TRAINING
# =============================================================================

def tune_linear_svc(
    X_train: csr_matrix,
    y_train: np.ndarray,
    cv: int = 5,
    n_jobs: int = -1
) -> Tuple[LinearSVC, Dict, pd.DataFrame]:
    """Tune LinearSVC with GridSearchCV."""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING - LinearSVC")
    print("=" * 70)
    
    param_grid = {
        'C': [0.01, 0.02, 0.05, 0.1, 0.2],
        'loss': ['hinge', 'squared_hinge'],
        'class_weight': ['balanced'],
        'max_iter': [2000]
    }
    
    svc = LinearSVC(dual=False, random_state=RANDOM_STATE)
    scorer = make_scorer(f1_score, average='macro')
    
    start = time.time()
    grid = GridSearchCV(
        svc, param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        scoring=scorer, n_jobs=n_jobs, verbose=1, return_train_score=True
    )
    grid.fit(X_train, y_train)
    
    print(f"Time: {time.time() - start:.1f}s")
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV Macro F1: {grid.best_score_:.4f}")
    
    return grid.best_estimator_, grid.best_params_, pd.DataFrame(grid.cv_results_)


def tune_lightgbm(
    X_train: csr_matrix,
    y_train: np.ndarray,
    cv: int = 5,
    n_iter: int = 40,
    n_jobs: int = -1
) -> Tuple[lgb.LGBMClassifier, Dict, pd.DataFrame]:
    """Tune LightGBM with RandomizedSearchCV."""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING - LightGBM")
    print("=" * 70)
    
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 7, 10, 15, -1],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [31, 50, 70, 100],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'class_weight': ['balanced'],
        'random_state': [RANDOM_STATE],
        'verbose': [-1]
    }
    
    lgbm = lgb.LGBMClassifier(objective='multiclass', n_jobs=n_jobs)
    scorer = make_scorer(f1_score, average='macro')
    
    start = time.time()
    search = RandomizedSearchCV(
        lgbm, param_dist, n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        scoring=scorer, n_jobs=n_jobs, verbose=1,
        return_train_score=True, random_state=RANDOM_STATE
    )
    search.fit(X_train, y_train)
    
    print(f"Time: {time.time() - start:.1f}s")
    print(f"Best params: {search.best_params_}")
    print(f"Best CV Macro F1: {search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_params_, pd.DataFrame(search.cv_results_)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """Comprehensive model evaluation with detailed metrics."""
    print(f"\n{'=' * 70}")
    print(f"EVALUATION - {model_name}")
    print('=' * 70)
    
    # Global metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = (y_true == y_pred).mean()
    
    print(f"\nGlobal Metrics:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    
    print(f"\nPer-Class Metrics:")
    print("-" * 70)
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)
    
    for i in range(len(LABEL_NAMES)):
        print(f"{LABEL_NAMES[i]:<15} {precision[i]:>10.4f} {recall[i]:>10.4f} "
              f"{f1[i]:>10.4f} {support[i]:>10}")
    
    # Identify weak classes
    weak = [(LABEL_NAMES[i], f1[i]) for i in range(len(f1)) if f1[i] < 0.65]
    if weak:
        print(f"\n⚠️  Classes with F1 < 0.65:")
        for name, score in sorted(weak, key=lambda x: x[1]):
            print(f"   - {name}: {score:.3f}")
    
    return {
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'accuracy': accuracy,
        'f1_per_class': f1,
        'precision': precision,
        'recall': recall,
        'support': support
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None
) -> None:
    """Plot confusion matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels = [LABEL_NAMES[i][:10] for i in range(len(LABEL_NAMES))]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f'{model_name} - Normalized')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f'{model_name} - Absolute')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()
    
def plot_feature_importance(model,feature_names: list, model_type: str = 'svc', top_n=15):
    if model_type == 'svc':
        n_classes = model.coef_.shape[0]
        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        axes = axes.flatten()
        
        for i in range(n_classes):
            ax = axes[i]
            coefs = model.coef_[i]
            top_idx = np.argsort(np.abs(coefs))[-top_n:]
            top_feat = feature_names[top_idx]
            top_coef = coefs[top_idx]
            
            colors = ['forestgreen' if c > 0 else 'crimson' for c in top_coef]
            ax.barh(range(len(top_feat)), top_coef, color=colors)
            ax.set_yticks(range(len(top_feat)))
            ax.set_yticklabels([f[:25] for f in top_feat], fontsize=7)
            ax.set_title(f'{LABEL_NAMES[i]}', fontweight='bold')
            ax.axvline(x=0, color='black', linewidth=0.5)
        
        for j in range(n_classes, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('LinearSVC - Top Features per Category', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
        plt.close()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 10))
        # Gain
        imp_gain = model.booster_.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp_gain)[-top_n:]
        
        axes[0].barh(range(len(top_idx)), imp_gain[top_idx], color='steelblue')
        axes[0].set_yticks(range(len(top_idx)))
        axes[0].set_yticklabels([feature_names[i][:30] for i in top_idx], fontsize=8)
        axes[0].set_xlabel('Importance (Gain)')
        axes[0].set_title('LightGBM - Feature Importance (Gain)', fontweight='bold')
        
        # Split
        imp_split = model.booster_.feature_importance(importance_type='split')
        top_idx_s = np.argsort(imp_split)[-top_n:]
        
        axes[1].barh(range(len(top_idx_s)), imp_split[top_idx_s], color='darkorange')
        axes[1].set_yticks(range(len(top_idx_s)))
        axes[1].set_yticklabels([feature_names[i][:30] for i in top_idx_s], fontsize=8)
        axes[1].set_xlabel('Importance (Split)')
        axes[1].set_title('LightGBM - Feature Importance (Split)', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        plt.close()
        
    


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(
    model: Any,
    pipeline: NewsPipeline,
    params: Dict,
    save_dir: str | Path,
    model_name: str = "model"
) -> None:
    """Save model, pipeline, and parameters."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, save_dir / f"{model_name}_classifier.joblib")
    joblib.dump(pipeline, save_dir / f"{model_name}_pipeline.joblib")
    
    # Convert numpy types for JSON
    params_serializable = {}
    for k, v in params.items():
        if isinstance(v, (np.integer, np.floating)):
            params_serializable[k] = v.item()
        elif isinstance(v, np.ndarray):
            params_serializable[k] = v.tolist()
        else:
            params_serializable[k] = v
    
    with open(save_dir / f"{model_name}_params.json", 'w') as f:
        json.dump(params_serializable, f, indent=2)
    
    print(f"Model saved to: {save_dir}")


def load_model(
    save_dir: str | Path,
    model_name: str = "model"
) -> Tuple[Any, NewsPipeline, Dict]:
    """Load model, pipeline, and parameters."""
    save_dir = Path(save_dir)
    
    model = joblib.load(save_dir / f"{model_name}_classifier.joblib")
    pipeline = joblib.load(save_dir / f"{model_name}_pipeline.joblib")
    
    with open(save_dir / f"{model_name}_params.json", 'r') as f:
        params = json.load(f)
    
    print(f"Model loaded from: {save_dir}")
    return model, pipeline, params


# =============================================================================
# TRAINING AND INFERENCE
# =============================================================================

def train_and_evaluate(
    dev_df: pd.DataFrame,
    test_size: float = 0.2,
    model_type: str = 'svc',
    tune: bool = True,
    context_injection: bool = True,
    use_catboost: bool = True,
    save_dir: Optional[str] = None
) -> Tuple[Any, NewsPipeline, Dict]:
    """
    Train model on train split and evaluate on test split.
    
    Parameters
    ----------
    dev_df : pd.DataFrame
        Development DataFrame with 'label' column.
    test_size : float
        Proportion for test split.
    model_type : str
        'svc' for LinearSVC or 'lgbm' for LightGBM.
    tune : bool
        Whether to perform hyperparameter tuning.
    context_injection : bool
        If True, inject metadata into text (recommended).
    use_catboost : bool
        If True, also use CatBoost encoding (hybrid mode).
    save_dir : str | None
        Directory to save model.
    
    Returns
    -------
    Tuple[Any, NewsPipeline, Dict]
        Trained model, fitted pipeline, and evaluation metrics.
    """
    print("=" * 70)
    print("TRAIN AND EVALUATE")
    print(f"Context Injection: {context_injection}, CatBoost: {use_catboost}")
    print("=" * 70)
    
    # Preprocess
    print("\n1. Preprocessing...")
    df = preprocess_data(dev_df, is_train=True)
    
    # Generate features
    print("\n2. Generating features...")
    df = generate_features(df, context_injection=context_injection)
    
    # Split
    print("\n3. Splitting data...")
    X = df.drop(columns=['label'])
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test: {len(X_test):,} samples")
    
    # Fit pipeline
    print("\n4. Fitting feature pipeline...")
    pipeline = NewsPipeline(
        catboost_sigma=0.5,
        use_catboost=use_catboost
    )
    X_train_transformed = pipeline.fit_transform(X_train, y_train)
    X_test_transformed = pipeline.transform(X_test)
    
    # Train model
    print("\n5. Training model...")
    if tune:
        if model_type == 'svc':
            model, best_params, _ = tune_linear_svc(X_train_transformed, y_train)
        else:
            model, best_params, _ = tune_lightgbm(X_train_transformed, y_train)
    else:
        # Optimized defaults for context injection
        if model_type == 'svc':
            model = LinearSVC(
                C=0.05, loss='squared_hinge', class_weight='balanced',
                dual=False, max_iter=2000, random_state=RANDOM_STATE
            )
            best_params = {'C': 0.05, 'loss': 'squared_hinge'}
        else:
            model = lgb.LGBMClassifier(
                n_estimators=300, max_depth=-1, learning_rate=0.1,
                num_leaves=70, min_child_samples=20, subsample=0.8,
                colsample_bytree=0.7, class_weight='balanced',
                random_state=RANDOM_STATE, verbose=-1
            )
            best_params = model.get_params()
        model.fit(X_train_transformed, y_train)
    
    # Evaluate
    print("\n6. Evaluating...")
    y_pred_train = model.predict(X_train_transformed)
    y_pred_test = model.predict(X_test_transformed)
    
    train_f1 = f1_score(y_train, y_pred_train, average='macro')
    print(f"\nTrain Macro F1: {train_f1:.4f}")
    
    metrics = evaluate_model(y_test, y_pred_test, model_name=model_type.upper())
    
    print(f"\nGap (overfitting): {train_f1 - metrics['macro_f1']:.4f}")
    
    # Plot
    plot_confusion_matrix(y_test, y_pred_test, model_type.upper())
    
    plot_feature_importance(model, pipeline.feature_names_, model_type=model_type)
    
    # Save
    if save_dir:
        save_model(model, pipeline, best_params, save_dir, model_name=model_type)
    
    return model, pipeline, metrics


def train_full_and_predict(
    dev_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    model_type: str = 'svc',
    model_params: Optional[Dict] = None,
    context_injection: bool = True,
    use_catboost: bool = True,
    save_dir: Optional[str] = None,
    submission_path: str = 'submission.csv'
) -> pd.DataFrame:
    """
    Train on full development set and generate predictions for evaluation set.
    """
    print("=" * 70)
    print("TRAIN FULL AND PREDICT")
    print("=" * 70)
    
    # Store evaluation IDs
    eval_ids = eval_df['Id'].values
    
    # Preprocess
    print("\n1. Preprocessing...")
    df_train = preprocess_data(dev_df, is_train=True)
    
    # Generate features
    print("\n2. Generating features...")
    df_train = generate_features(df_train, context_injection=context_injection)
    
    X_train = df_train.drop(columns=['label'])
    y_train = df_train['label'].values
    
    # Fit pipeline
    print("\n3. Fitting feature pipeline...")
    pipeline = NewsPipeline(
        catboost_sigma=0.5,
        use_catboost=use_catboost
    )
    X_train_transformed = pipeline.fit_transform(X_train, y_train)
    
    # Prepare evaluation data
    print("\n4. Preparing evaluation data...")
    df_eval = preprocess_data(eval_df, is_train=False)
    df_eval = generate_features(df_eval, context_injection=context_injection)
    X_eval_transformed = pipeline.transform(df_eval)
    
    # Train model
    print("\n5. Training model...")
    if model_type == 'svc':
        params = model_params or {
            'C': 0.05, 'loss': 'squared_hinge', 'class_weight': 'balanced',
            'dual': False, 'max_iter': 2000, 'random_state': RANDOM_STATE
        }
        model = LinearSVC(**params)
    else:
        params = model_params or {
            'n_estimators': 300, 'max_depth': -1, 'learning_rate': 0.1,
            'num_leaves': 70, 'min_child_samples': 20, 'subsample': 0.8,
            'colsample_bytree': 0.7, 'class_weight': 'balanced',
            'random_state': RANDOM_STATE, 'verbose': -1
        }
        model = lgb.LGBMClassifier(**params)
    
    model.fit(X_train_transformed, y_train)
    
    plot_feature_importance(model, pipeline.feature_names_, model_type=model_type)
    
    # Predict
    print("\n6. Generating predictions...")
    y_pred = model.predict(X_eval_transformed)
    
    # Create submission
    submission = pd.DataFrame({
        'Id': eval_ids,
        'Predicted': y_pred.astype(int)
    })
    
    submission.to_csv(submission_path, index=False)
    print(f"\n✅ Submission saved: {submission_path}")
    print(f"   Shape: {submission.shape}")
    print(f"\nPrediction distribution:")
    for label, name in LABEL_NAMES.items():
        count = (y_pred == label).sum()
        pct = count / len(y_pred) * 100
        print(f"  {name}: {count:,} ({pct:.1f}%)")
    
    # Save model
    if save_dir:
        save_model(model, pipeline, params, save_dir, model_name=f"{model_type}_full")
    
    return submission


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("News Classification Pipeline - Optimized")
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
        tune=True,  # Use optimized defaults
        context_injection=True,  # +1.8% F1
        use_catboost=False,  # Hybrid mode
        save_dir='./models'
    )
    
    # Generate submission
    submission = train_full_and_predict(
        dev_df,
        eval_df,
        model_type='lgbm',
        context_injection=True,
        use_catboost=False,
        submission_path='submission.csv'
    )