import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from pathlib import Path
import ast
import yaml

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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils._set_output import _SetOutputMixin
# LightGBM
import lightgbm as lgb

# Scipy
from scipy.sparse import hstack, csr_matrix

# Category encoders
from category_encoders import CatBoostEncoder

import re

from dataclasses import dataclass, field

from typing import Any, Tuple, Dict, Union, List



warnings.filterwarnings('ignore')
np.random.seed(42)




with open("config/config.yaml", 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)
    
LABEL_NAMES = config['LABEL_NAMES']

RANDOM_STATE = config['RANDOM_STATE']


# ====================================================================================================
# DATA LOADING AND BASIC PREPROCESSING
# ====================================================================================================

def load_file(
    dev_path: str | Path,
    evl_path: str | Path | None= None,
    **args) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Load the development and evaluation datasets from CSV files.
    
    Paramters
    ---------
    dev_path : str | Path
        Path to the development dataset CSV file.
    evl_path : str | Path | None, optional
        Path to the evaluation dataset CSV file. Default is None.
    **args
        Additional arguments to pass to pd.read_csv().
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame | None]
        A tuple containing the development DataFrame and the evaluation DataFrame (or None if not provided).
    """
    
    
    dev_df = pd.read_csv(dev_path, **args)
    print(f"  Development set: {dev_df.shape[0]:,} samples, {dev_df.shape[1]} features")
    
    if evl_path is not None:
        evl_df = pd.read_csv(evl_path, **args)
        print(f"Evaluation data shape: {evl_df.shape}")

    
    return dev_df, evl_df if evl_path is not None else None



def preprocessing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input DataFrame by handling duplicates and missing values:
    - Converts 'timestamp' column to datetime.
    - Drops duplicates based on 'source', 'title', 'article', 'label', keeping the most recent one.
    - Drops duplicates that have the same 'source', 'title', 'article' but different 'label'.
    - Drops 'Id' column if present.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to preprocess.
        
    """
    
    
    temp_df = df.copy()
    original_len = temp_df.shape[0]
    
    # Converting timestamp to datetime
    print("Converting timestamp to datetime...")
    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], errors='coerce')
    
    # Dropping duplicates based on source, title, article, label keeping the most recent one
    print("Dropping duplicates based on source, title, article, label keeping the most recent one...")
    temp_df = temp_df.sort_values(by='timestamp', ascending=False, na_position='last')
    temp_df = temp_df.drop_duplicates(subset = ['source','title','article','label'], keep='first')
    print(f" {original_len - temp_df.shape[0]:,} samples removed")
    original_len = temp_df.shape[0]
    
    # Dropping duplicates that have the same source, title, article but different label. This is to avoid ambiguity
    print("Dropping duplicates that have the same source, title, article but different label...")
    counts = temp_df.groupby(['source', 'title', 'article'])['label'].transform('nunique')
    temp_df = temp_df[counts == 1]
    print(f" {original_len - temp_df.shape[0]:,} samples removed")
    
    
    # Dropping id column
    print("Dropping id column...")
    if 'Id' in temp_df.columns:
        temp_df = temp_df.drop(columns=['Id'])
        
    temp_df = temp_df.sample(frac=1, random_state=RANDOM_STATE) 
    temp_df = temp_df.reset_index(drop=True)
    
    print(f"  Preprocessed Data: {temp_df.shape[0]:,} samples, {temp_df.shape[1]} features")
    
    return temp_df
    


def clean_text(text: str) -> str:
    """
    Clean and preprocess a single text string.
    
    Parameters
    ----------
    text : str
        The input text to clean.
    
    Returns
    -------
    str
        The cleaned and preprocessed text string.
    
    Notes
    -----
    This function performs the following preprocessing steps:
    1. Handles NaN/None values by returning an empty string.
    2. Converts text to lowercase.
    3. Removes HTML entities (e.g., &#39; -> ', &amp; -> &).
    4. Removes HTML tags.
    5. Removes HTML entities
    6. Removes URLs (http, https, www).
    7. Removes special characters, keeping only alphanumeric characters and spaces.
    8. Removes extra whitespace (converts multiple spaces to a single space).
    9. Removes standalone numbers.
    10. Removes naive words like said, says, say, year, years.
    """
    
    # 1. Handle NaN/None values
    if pd.isna(text) or text is None:
        return ""
    
    # 2. Convert text to string in case it is not already a string
    text = str(text)
    
    # 3. Convert to lowercase for uniformity
    text = text.lower()
    
    # 4. Remove HTML entities (e.g., &#39; -> ', &amp; -> &)
    text = re.sub(r'&#?\w+;', ' ', text)
    
    # 5. Remove common HTML tags like <a>, <div>, <span>, etc.
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 6. Remove URLs (http, https, www)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # 7. Remove special characters, keeping only alphanumeric characters and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\b(alt|img|src|href|title|width|height|http|https|www)\b', ' ', text)
    
    # 8. Remove extra whitespace (convert multiple spaces into a single space)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 9. Remove standalone numbers
    text =  re.sub(r'\b\d+\b', '', text)
    
    # 10. Remove naive words like said, says, say, 
    text = re.sub(r'\b(said|says|say|year|years)\b', '', text)
    
    return text




#=====================================================================================================
# TEXT MANIPULATION
#=====================================================================================================


def extract_title_suffix(title: str) -> str | None:
    """
    Some titles have suffixes in brackets that provide additional context. This function extracts such suffixes from the title string.
    
    Parameters
    ----------
    title : str
        The title string to extract suffix from.
        
    Returns
    -------
    str | None
        The extracted suffix if present, otherwise None.
    
    """
    
    if isinstance(title, float):
        return None
    
    # Find words between brackets
    suffix_pattern = r'\(([^)]+)\)\s*$' #$ is usefull to find only at the end of the string
    match = re.search(suffix_pattern, title)
    return match.group(1) if match else None

def extract_first_domain(article: str) -> str:
    """  
    Some urls are embedded in the article text. This function extracts the domain of the first URL found in the article string.
    
    Parameters
    ----------
    article : str
        The article string to extract the first URL domain from.
        
    Returns
    -------
    str | None
        The extracted domain if a URL is found, otherwise None.
    
    """
    if pd.isna(article):
        return None
    match = re.search(r'https?://([^\s<>"{}|\\^`\[\]/]+)', str(article))
    if match:
        domain = match.group(1).replace('www.', '')
        return domain.split('/')[0]
    return None

def count_links(article: str) -> Dict[str, int]:
    """
    Count the number of links per type (image, ads, feeds) in the article string.
    
    Parameters
    ----------
    article : str
        The article string to analyze.
        
    Returns
    -------
    dict
        A dictionary with counts of links per type: {'n_links': int, 'n_images': int, 'n_ads': int, 'n_feeds': int}.
    
    """
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

RSS_TO_LABEL = {
    # International (label 0)
    'world': 0,
    'europe': 0,
    'politics': 0,
    'elections': 0,
    'us': 0,
    
    # Business (label 1)
    'business': 1,
    
    # Technology (label 2)
    'tech': 2,
    'science': 2,
    
    # Entertainment (label 3)
    'entertainment': 3,
    
    # Sports (label 4)
    'sports': 4,
    
    # General (label 5)
    'cnn_topstories': 5,
    
    # Health (label 6)
    'health': 6,
}



# =============================================================================
# FUNZIONI DI ESTRAZIONE
# =============================================================================

def extract_rss_label(text):
    """
    Estrae la label predetta dalla categoria RSS presente nel testo.
    
    Questa feature ha 100% di accuratezza ma copre solo ~9% degli articoli.
    
    Parameters
    ----------
    text : str
        Testo dell'articolo (colonna 'article')
    
    Returns
    -------
    int or None
        Label predetta (0-6) se trova RSS category, None altrimenti
    """
    if pd.isna(text) or not isinstance(text, str):
        return None
    
    # Pattern per estrarre la categoria RSS
    match = re.search(r'/rss/([a-z_]+)', text.lower())
    if match:
        category = match.group(1).split('?')[0]  # Rimuovi query string
        return RSS_TO_LABEL.get(category, None)
    return None


def extract_all_links(text):
    """
    Estrae tutti i link da un testo, inclusi quelli annidati (separati da *).
    
    Gestisce:
    1. Link standard http/https
    2. Link dopo href= o src=
    3. Link annidati separati da *
    
    Parameters
    ----------
    text : str
        Testo dell'articolo
    
    Returns
    -------
    list
        Lista di URL estratti
    """
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    links = []
    
    # Pattern 1: Link standard http/https
    for match in re.findall(r'https?://[^\s<>"\')\]]+', text, re.IGNORECASE):
        links.append(match)
    
    # Pattern 2: href="..." o src="..."
    for match in re.findall(r'(?:href|src)=["\']([^"\']+)["\']', text, re.IGNORECASE):
        if 'http' in match.lower():
            links.append(match)
    
    # Separa link annidati (separati da *)
    all_links = []
    for link in links:
        if '*http' in link:
            parts = link.split('*http')
            all_links.append(parts[0])
            for part in parts[1:]:
                all_links.append('http' + part)
        else:
            all_links.append(link)
    
    return all_links


def extract_link_info(url):
    """
    Estrae informazioni strutturate da un URL.
    
    Parameters
    ----------
    url : str
        URL da analizzare
    
    Returns
    -------
    dict
        Dizionario con: domain, domain_main, path, path_parts, query
    """
    try:
        # Pulisci URL
        url = re.sub(r'[<>"\')\]\s]+$', '', url)
        parsed = urlparse(url)
        
        domain = parsed.netloc.split(':')[0] if parsed.netloc else ''
        domain_parts = domain.split('.')
        domain_main = '.'.join(domain_parts[-2:]) if len(domain_parts) >= 2 else domain
        
        return {
            'full_url': url,
            'domain': domain,
            'domain_main': domain_main,
            'path': parsed.path,
            'path_parts': [p for p in parsed.path.split('/') if p],
            'query': parsed.query
        }
    except:
        return None


def extract_link_features(text):
    """
    Estrae tutte le feature derivate dai link per un articolo.
    
    Parameters
    ----------
    text : str
        Testo dell'articolo
    
    Returns
    -------
    dict
        Dizionario con:
        - rss_label: label predetta da RSS (o None)
        - has_links: bool
        - num_links: int
        - has_yahoo_link: bool
        - has_reuters_link: bool
        - has_cnn_link: bool
        - has_img: bool
        - main_domain: dominio principale (o None)
    """
    features = {
        'rss_label': None,
        'has_links': False,
        'num_links': 0,
        'has_yahoo_link': False,
        'has_reuters_link': False,
        'has_cnn_link': False,
        'has_img': False,
        'main_domain': None
    }
    
    if pd.isna(text) or not isinstance(text, str):
        return features
    
    text_lower = text.lower()
    
    # Feature principale: RSS label
    features['rss_label'] = extract_rss_label(text)
    
    # Estrai link
    links = extract_all_links(text)
    features['has_links'] = len(links) > 0
    features['num_links'] = len(links)
    
    # Presenza di specifici domini
    features['has_yahoo_link'] = 'yahoo.com' in text_lower
    features['has_reuters_link'] = 'reuters.com' in text_lower
    features['has_cnn_link'] = 'cnn.com' in text_lower
    features['has_img'] = '<img' in text_lower
    
    # Dominio principale (primo link)
    if links:
        info = extract_link_info(links[0])
        if info:
            features['main_domain'] = info['domain_main']
    
    return features


def add_link_features(df, article_col='article'):
    """
    Aggiunge le feature estratte dai link a un DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con gli articoli
    article_col : str
        Nome della colonna con il testo degli articoli
    
    Returns
    -------
    pd.DataFrame
        DataFrame con le nuove colonne di feature
    """
    # Estrai feature per ogni articolo
    link_features = df[article_col].apply(extract_link_features)
    
    # Espandi in colonne separate
    features_df = pd.DataFrame(link_features.tolist())
    
    # Aggiungi al DataFrame originale
    result = pd.concat([df, features_df], axis=1)
    
    return result



def generate_features(
    df: pd.DataFrame, 
    config: dict,
    article_bins: list  = [0, 10, 100, 180, np.inf],
    article_labels: list = [0,1,2,3],
    title_weight: int = 2,
    article_weight: int = 1) -> pd.DataFrame:

    temp_df = df.copy()
    

    
    # Title suffix
    if config['FEATURES']['EXTRACT_TITLE_SUFFIX']:
        temp_df['title_suffix'] = temp_df['title'].apply(extract_title_suffix)
    
    # Domain
    if config['FEATURES']['EXTRACT_DOMAIN']:
        temp_df['first_link_domain'] = temp_df['article'].apply(extract_first_domain)
        
    if config['FEATURES']['LINKS_FEATURES']:
        temp_df = add_link_features(temp_df, 'article')
    
    # Link counts
    if config['FEATURES']['EXTRACT_LINKS']:
        link_counts = temp_df['article'].apply(count_links).apply(pd.Series)
        temp_df = pd.concat([temp_df, link_counts], axis=1)
    
    # Article length
    temp_df['article_length_num'] = temp_df['article'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    temp_df['article_length'] = pd.cut(
        temp_df['article_length_num'],
        bins=article_bins,
        labels=article_labels
    ).astype(float).fillna(1)
    temp_df = temp_df.drop(columns=['article_length_num'])
    
    # Combined text
    temp_df['combined_text'] = (
        (temp_df['title'].apply(clean_text) + " ") * title_weight +
        (" " + temp_df['article'].apply(clean_text)) * article_weight
    )
    
    return temp_df


class FullPipeline(BaseEstimator, TransformerMixin):
    
    def __init__(self, config: dict):
        self.config = config
        self.tfidf = None
        self.scaler = None
        self.feature_names_ = None
        self.num_cols_complete = [
        'page_rank', 'n_links', 'n_images', 'n_ads', 'n_feeds', 'article_length', 'rss_label','num_links']
        self.cat_cols_complete = ['source', 'first_link_domain', 'title_suffix']
        
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, np.ndarray]:
        
        features_list = []
        feature_names = []
        
        # 1. TF-IDF
        if self.config['TFIDF']['ENABLE']:
            tfidf_params = self.config['TFIDF']['PARAMS']
            ng = self.config['TFIDF']['PARAMS'].get('ngram_range')
            if isinstance(ng, str):
                ng = ast.literal_eval(ng)
            if isinstance(ng, list):
                ng = tuple(ng)
            tfidf_params['ngram_range'] = ng
            

            
            try:
                tfidf_params['token_pattern'] = ast.literal_eval(self.config['TFIDF']['PARAMS']['token_pattern'])
            except:
                tfidf_params['token_pattern']  = r'(?u)\b[a-zA-Z]{3,}\b'
            
            self.tfidf = TfidfVectorizer(**tfidf_params)
            X_tfidf = self.tfidf.fit_transform(X['combined_text'])
            features_list.append(X_tfidf)
            feature_names.extend([f"tfidf_{w}" for w in self.tfidf.get_feature_names_out()])
            print(f"  TF-IDF: {X_tfidf.shape[1]} features")
        
        # 2. Numerical features
        num_cols = [c for c in self.num_cols_complete if c in X.columns]
        if num_cols:
            self.scaler = StandardScaler()
            X_num = self.scaler.fit_transform(X[num_cols].fillna(0))
            features_list.append(csr_matrix(X_num))
            feature_names.extend([f"num_{c}" for c in num_cols])
            print(f"  Numerical: {len(num_cols)} features")
        
        # 3. CatBoost encoding
        if self.config['CATBOOST']['ENABLE']:
            cat_cols = [c for c in self.cat_cols_complete if c in X.columns]
            if cat_cols:
                self.cat_encoders = {}
                cat_features = []
                
                for cls in np.unique(y):
                    y_binary = (y == cls).astype(int)
                    enc = CatBoostEncoder(cols=cat_cols, sigma=self.config['CATBOOST']['PARAMS']['sigma'])
                    X_enc = enc.fit_transform(X[cat_cols].fillna('missing'), y_binary)
                    self.cat_encoders[cls] = enc
                    cat_features.append(X_enc.values)
                    feature_names.extend([f"{c}_prob_cl{cls}" for c in cat_cols])
                
                X_cat = np.hstack(cat_features)
                features_list.append(csr_matrix(X_cat))
                print(f"  CatBoost: {X_cat.shape[1]} features")
        
        # 4. Timestamp cyclic features 
        if self.config['FEATURES']['CYCLE_TIME'] and 'timestamp' in X.columns:
            ts = pd.to_datetime(X['timestamp'], errors='coerce')
            self.median_ts = ts.median()
            ts = ts.fillna(self.median_ts)
            
            cyclic_features = pd.DataFrame({
                'is_ts_missing': pd.to_datetime(X['timestamp'], errors='coerce').isna().astype(int),
                'hour_sin': np.sin(2 * np.pi * ts.dt.hour / 24),
                'hour_cos': np.cos(2 * np.pi * ts.dt.hour / 24),
                'day_sin': np.sin(2 * np.pi * ts.dt.dayofweek / 7),
                'day_cos': np.cos(2 * np.pi * ts.dt.dayofweek / 7),
                'month_sin': np.sin(2 * np.pi * (ts.dt.month - 1) / 12),
                'month_cos': np.cos(2 * np.pi * (ts.dt.month - 1) / 12),
            })
            
            features_list.append(csr_matrix(cyclic_features.values))
            feature_names.extend(cyclic_features.columns.tolist())
            print(f"  Cyclic timestamp: {cyclic_features.shape[1]} features")
        
        
        X_combined = hstack(features_list)
        self.feature_names_ = np.array(feature_names)
        print(f"  TOTAL: {X_combined.shape[1]} features")
        
        return X_combined, y.values
    
    def transform(self, X: pd.DataFrame) -> Any:
        features_list = []
        
        # TF-IDF
        if self.config['TFIDF']['ENABLE'] and self.tfidf is not None:
            X_tfidf = self.tfidf.transform(X['combined_text'])
            features_list.append(X_tfidf)
        
        # Numerical
        num_cols = [c for c in self.num_cols_complete if c in X.columns]
        if num_cols and self.scaler is not None:
            X_num = self.scaler.transform(X[num_cols].fillna(0))
            features_list.append(csr_matrix(X_num))
        
        # CatBoost
        if self.config['CATBOOST']['ENABLE'] and hasattr(self, 'cat_encoders'):
            cat_cols = [c for c in self.cat_cols_complete if c in X.columns]
            if cat_cols:
                cat_features = []
                for cls, enc in self.cat_encoders.items():
                    X_enc = enc.transform(X[cat_cols].fillna('missing'))
                    cat_features.append(X_enc.values)
                X_cat = np.hstack(cat_features)
                features_list.append(csr_matrix(X_cat))
        
        # Timestamp
        if self.config['FEATURES']['CYCLE_TIME'] and 'timestamp' in X.columns:
            ts = pd.to_datetime(X['timestamp'], errors='coerce')
            ts = ts.fillna(self.median_ts)
            
            cyclic_features = pd.DataFrame({
                'is_ts_missing': pd.to_datetime(X['timestamp'], errors='coerce').isna().astype(int),
                'hour_sin': np.sin(2 * np.pi * ts.dt.hour / 24),
                'hour_cos': np.cos(2 * np.pi * ts.dt.hour / 24),
                'day_sin': np.sin(2 * np.pi * ts.dt.dayofweek / 7),
                'day_cos': np.cos(2 * np.pi * ts.dt.dayofweek / 7),
                'month_sin': np.sin(2 * np.pi * (ts.dt.month - 1) / 12),
                'month_cos': np.cos(2 * np.pi * (ts.dt.month - 1) / 12),
            })
            features_list.append(csr_matrix(cyclic_features.values))
        
        return hstack(features_list)
    

def tune_linear_svc(X_train, y_train, cv=5):
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING - LinearSVC")
    print("="*70)
    
    param_grid = {
        'C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
        'loss': ['hinge', 'squared_hinge'],
        'class_weight': ['balanced'],
        'max_iter': [2000]
    }
    
    svc = LinearSVC(dual=False, random_state=42)
    scorer = make_scorer(f1_score, average='macro')
    
    total_comb = np.prod([len(v) for v in param_grid.values()])
    print(f"\nGrid Search: {total_comb} combinations")
    
    start = time.time()
    grid = GridSearchCV(
        svc, param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=scorer, n_jobs=-1, verbose=1, return_train_score=True
    )
    grid.fit(X_train, y_train)
    
    print(f"\Time: {time.time()-start:.1f}s")
    
    results_df = pd.DataFrame(grid.cv_results_).sort_values('rank_test_score')
    
    print("\n--- TOP 5 CONFIGURATIONS ---")
    for i, row in results_df.head().iterrows():
        print(f"  Rank {int(row['rank_test_score'])}: C={row['param_C']}, loss={row['param_loss']} "
              f"-> CV F1={row['mean_test_score']:.4f} (+/-{row['std_test_score']:.4f})")
    
    print(f"\n✅ BEST: {grid.best_params_}")
    print(f"✅ BEST CV Macro F1: {grid.best_score_:.4f}")
    
    return grid.best_estimator_, grid.best_params_, results_df


def tune_lightgbm(X_train, y_train, cv=5):
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING - LightGBM")
    print("="*70)
    
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 7, 10, 15, -1],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [31, 50, 70, 100],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'class_weight': ['balanced'],
        'random_state': [42],
        'verbose': [-1]
    }
    
    lgbm = lgb.LGBMClassifier(objective='multiclass', n_jobs=-1)
    scorer = make_scorer(f1_score, average='macro')
    
    n_iter = 40
    print(f"\nRandomized Search: {n_iter} iterations")
    
    start = time.time()
    random_search = RandomizedSearchCV(
        lgbm, param_dist,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=scorer, n_jobs=-1, verbose=1,
        return_train_score=True, random_state=42
    )
    random_search.fit(X_train, y_train)
    
    print(f"\Time: {time.time()-start:.1f}s")
    
    results_df = pd.DataFrame(random_search.cv_results_).sort_values('rank_test_score')
    
    print("\n--- TOP 5 CONFIGURATIONS ---")
    for i, row in results_df.head().iterrows():
        params = row['params']
        print(f"  Rank {int(row['rank_test_score'])}: n_est={params.get('n_estimators')}, "
              f"depth={params.get('max_depth')}, lr={params.get('learning_rate')} "
              f"-> CV F1={row['mean_test_score']:.4f}")
    
    print(f"\n✅ BEST: {random_search.best_params_}")
    print(f"✅ BEST CV Macro F1: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_, results_df


def detailed_report(y_true, y_pred, model_name: str):
    print(f"\n{'='*70}")
    print(f"DETAILED ANALYSIS - {model_name}")
    print('='*70)
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = (y_true == y_pred).mean()
    
    print(f"\n{'GLOBAL METRICS':^40}")
    print("-"*40)
    print(f"{'Accuracy:':<25} {accuracy:.4f}")
    print(f"{'Macro F1:':<25} {macro_f1:.4f}")
    print(f"{'Weighted F1:':<25} {weighted_f1:.4f}")
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    
    print(f"\n{'METRICS PER CLASS':^70}")
    print("-"*70)
    print(f"{'Classe':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10} {'%':>8}")
    print("-"*70)
    
    total = support.sum()
    for i in range(len(LABEL_NAMES)):
        pct = support[i] / total * 100
        print(f"{LABEL_NAMES[i]:<15} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10} {pct:>7.1f}%")
    
    print("-"*70)
    print(f"{'MACRO AVG':<15} {precision.mean():>10.4f} {recall.mean():>10.4f} {f1.mean():>10.4f}")
    
    # Problemi
    print(f"\n{'ISSUES ANALYSIS':^40}")
    print("-"*40)
    weak = [(LABEL_NAMES[i], f1[i]) for i in range(len(f1)) if f1[i] < 0.65]
    if weak:
        print("⚠️  Classes with F1 < 0.65:")
        for name, score in sorted(weak, key=lambda x: x[1]):
            print(f"   - {name}: {score:.3f}")
    
    return {
        'macro_f1': macro_f1, 'weighted_f1': weighted_f1, 'accuracy': accuracy,
        'f1_per_class': f1, 'precision': precision, 'recall': recall, 'support': support
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels = [LABEL_NAMES[i][:10] for i in range(len(LABEL_NAMES))]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f'{model_name} - Normalized')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f'{model_name} - Absolute')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_f1_comparison(results: dict, save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(LABEL_NAMES))
    width = 0.35
    colors = ['steelblue', 'darkorange']
    
    for i, (name, metrics) in enumerate(results.items()):
        offset = (i - 0.5) * width
        ax.bar(x + offset, metrics['f1_per_class'], width, 
               label=f"{name} (Macro={metrics['macro_f1']:.3f})", color=colors[i])
    
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score per Category Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_NAMES[i] for i in range(len(LABEL_NAMES))], rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0.65, color='red', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_feature_importance_svc(model, feature_names, top_n=15, save_path=None):
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
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_feature_importance_lgbm(model, feature_names, top_n=30, save_path=None):
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
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_cv_tuning(results_df, param_name, model_name, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    results_df = results_df.copy()
    results_df['param_val'] = results_df['params'].apply(lambda x: x[param_name])
    results_df = results_df.dropna(subset=['param_val'])
    
    grouped = results_df.groupby('param_val').agg({
        'mean_test_score': 'mean',
        'std_test_score': 'mean'
    }).reset_index()
    
    ax.errorbar(range(len(grouped)), grouped['mean_test_score'],
                yerr=grouped['std_test_score'], fmt='o-', capsize=5, capthick=2, color='steelblue')
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped['param_val'], rotation=45)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Macro F1')
    ax.set_title(f'{model_name} - Effect of {param_name}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()












def apply_title_extraction(df: pd.DataFrame) -> pd.DataFrame:
    
    temp_df = df.copy()
    
    temp_df['title_suffix'] = temp_df['title'].apply(extract_title_suffix)
    return temp_df


#=====================================================================================================
# LINKS/DOMAINS EXTRACTION
#=====================================================================================================
def extract_link(article: str, first_only=False):
    if pd.isna(article) or article is None:
        return None

    if not isinstance(article, str):
        article = str(article)
    
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    
    match = re.findall(url_pattern, article)
    
    if first_only:
        return match[0] if match else None
    return match if match else None

def extract_domain(link: str):
    if pd.isna(link) or link is None:
        return None
    parsed = urlparse(link)
    domain = parsed.netloc.replace('www.', '')
    return domain 

def categorize_link(link: str):
    link_lower = link.lower()
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.ico']
    image_domains = ['yimg.com', 'img.', 'images.', 'static.']
    feed_patterns = ['feeds.', 'rss.', 'feed.', 'feedburner']
    ad_domains = ['doubleclick', 'adlog', 'pheedo', 'ad.']
    
    # images
    for ext in image_extensions:
        if ext in link_lower:
            return 'image'
    
    # Images
    for img_dom in image_domains:
        if img_dom in link_lower:
            return 'image'
    
    # Feed RSS
    for feed in feed_patterns:
        if feed in link_lower:
            return 'feed'
    
    # Advertising
    for ad in ad_domains:
        if ad in link_lower:
            return 'ad'
    
    return 'content'

def apply_domain_extraction(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df.copy()
    
    temp_df['first_link_domain'] = temp_df['article'].apply(lambda row: extract_link(row, first_only=True)).apply(extract_domain) 
    return temp_df


def add_links_numerics(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df.copy()
    
    temp_df['n_links'] = temp_df['article'].apply(lambda x: len(extract_link(x)) if extract_link(x) is not None else 0)
    temp_df['n_images'] = temp_df['article'].apply(lambda x: len([link for link in extract_link(x) if categorize_link(link) == 'image']) if extract_link(x) is not None else 0)
    temp_df['n_ads'] = temp_df['article'].apply(lambda x: len([link for link in extract_link(x) if categorize_link(link) == 'ad']) if extract_link(x) is not None else 0)
    temp_df['n_feeds'] = temp_df['article'].apply(lambda x: len([link for link in extract_link(x) if categorize_link(link) == 'feed']) if extract_link(x) is not None else 0)
    
    return temp_df

#=====================================================================================================
# ADD IS_LABEL_FEATURE COLUMNS
#=====================================================================================================


def top_feature_by_category(df: pd.DataFrame, top_n: int = 5, feature: str = "title_suffix", threshold: float = 0.95) -> pd.DataFrame:
    """"
    For each label, compute the top n values of the given 'feature' above a certain 'threshold'
    
    Paramters:
    ----------
    top_n : int, optional
        The number of top values to consider for each label (default is 5) for the given 'feature'.
    feature : str, optional
        The feature/column name to analyze (chosen from ['title_suffix', 'first_link_domain']) (default is 'title_suffix').
    threshold : float, optional
        The minimum proportion threshold for a value of a feature to be considered (default is 0.95).
        
    Returns:
    -------
    dict
        A dictionary where keys are labels and values are lists of top n feature values.
        
    """
    
    
    if feature == "title_suffix":
        if 'title_suffix' not in df.columns:
            df = apply_title_extraction(df)
    elif feature == "first_link_domain":
        if 'first_link_domain' not in df.columns:
            df = apply_domain_extraction(df)
    elif feature == 'source':
        pass
    else:
        raise ValueError("Feature must be one of ['title_suffix', 'first_link_domain', 'source']")
    
    matrix = df.groupby(['label', f'{feature}']).size().unstack(fill_value=0)
    relative_matrix = (matrix / matrix.sum(axis=0))
    
    mask_matrix = matrix >= 5
    mask_relative = relative_matrix > threshold
    mask_combined = mask_relative & mask_matrix
    
    top_n_title_suffixes_per_label = {}

    for label in mask_combined.index:
        valid_title_suffixes = mask_combined.loc[label][mask_combined.loc[label] == True].index.tolist()
        
        if valid_title_suffixes:
            sorted_title_suffixes = relative_matrix.loc[label, valid_title_suffixes].sort_values(ascending=False).head(top_n)
            top_n_title_suffixes_per_label[label] = sorted_title_suffixes.index.tolist()
        
    return top_n_title_suffixes_per_label


def add_is_label_feature_columns(
    df: pd.DataFrame,
    dict_top_values: dict,
    feature: str = "title_suffix",
    keep_column: bool = True) -> pd.DataFrame:
    
    """
    For each label, compute the top n values of the given 'feature' above a certain 'threshold' and create binary columns indicating presence of these features.


    Parameters
    ----------
    
    df : pd.DataFrame
        The input DataFrame containing the data.
    dict_top_values : dict
        A dictionary where keys are labels and values are lists of top n feature values.
    feature : str, optional
        The feature/column name to analyze (chosen from ['title_suffix', 'first_link_domain']) (default is 'title_suffix').
    keep_column : bool, optional
        Whether to keep the original feature column in the DataFrame (default is True).
        

    
    Returns:
    -------
    pd.DataFrame
        The DataFrame with additional binary columns for each top feature per label.
    """
    
    df = df.copy()
    
    if feature == "title_suffix":
        if 'title_suffix' not in df.columns:
            df = apply_title_extraction(df)
    elif feature == "first_link_domain":
        if 'first_link_domain' not in df.columns:
            df = apply_domain_extraction(df)
    elif feature == 'source':
        pass
    else:
        raise ValueError("Feature must be one of ['title_suffix', 'first_link_domain, 'source]")
    
    

    
    for key, value in dict_top_values.items():
        col_name = f'is_{str(key).lower()}_{feature}'
        df[col_name] = df[feature].isin(value).astype(int)
        
    if not keep_column:
        df.drop(columns=[feature], inplace=True)
    
    return df
    
    
#=============================================================================================
# FEATURES ENG
#=============================================================================================

def compute_article_length(df: pd.DataFrame, bins: list | None = None, labels: list | None = None) -> pd.DataFrame:
    """ 
    Compute the binned length of an article based on word count.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the article text.
        
    Returns
    -------
    pd.DataFrame
        The DataFrame with an additional column 'article_length' indicating the length category.
    """
    
    temp_df = df.copy()
    
    temp_df['article_length_num'] = temp_df['article'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    
    if bins is None or labels is None:
        bins = [0, 10, 100, 180, np.inf]
        labels = [0, 1, 2, 3]
    temp_df['article_length'] = pd.cut(temp_df['article_length_num'], bins=bins, labels=labels)
    temp_df.drop(columns=['article_length_num'], inplace=True)

    return temp_df

"""
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df.copy()
    
    temp_df['year'] = temp_df['timestamp'].dt.year.astype(int)
    temp_df['month'] = temp_df['timestamp'].dt.month
    temp_df['hour'] = temp_df['timestamp'].dt.hour
    temp_df['dayofweek'] = temp_df['timestamp'].dt.dayofweek
    temp_df['week'] = temp_df['timestamp'].dt.isocalendar().week
    
    temp_df['sin_month'], temp_df['cos_month'] = np.sin(2 * np.pi * (temp_df['month']-1) / 12), np.cos(2 * np.pi * (temp_df['month']-1) / 12)
    temp_df['sin_hour'], temp_df['cos_hour'] = np.sin(2 * np.pi * temp_df['hour'] / 24), np.cos(2 * np.pi * temp_df['hour'] / 24)
    temp_df['sin_dayofweek'], temp_df['cos_dayofweek'] = np.sin(2 * np.pi * temp_df['dayofweek'] / 7), np.cos(2 * np.pi * temp_df['dayofweek'] / 7)
    temp_df['sin_week'], temp_df['cos_week'] = np.sin(2 * np.pi * (temp_df['week'] -1 ) / 52), np.cos(2 * np.pi * (temp_df['week'] -1 ) / 52)
    
    
    temp_df = temp_df.drop(columns=['month', 'hour', 'dayofweek', 'week', 'timestamp'])
    return temp_df
"""
    
    


#=====================================================================================================
# TRANSFORMERS
#=====================================================================================================`
@dataclass
class FeatureGenerator(BaseEstimator, TransformerMixin, _SetOutputMixin):
    
    def get_feature_names_out(self, input_features=None):
        return np.array(['source', 'title', 'article', 'page_rank', 'timestamp', 
                        'title_suffix', 'first_link_domain', 'n_links', 
                     'n_images', 'n_ads', 'n_feeds', 'article_length'])
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_out = X.copy()

        X_out = apply_title_extraction(X_out)
        X_out = apply_domain_extraction(X_out)
   
   
        X_out = add_links_numerics(X_out)
   
        X_out = compute_article_length(X_out)
        return X_out
    
    def fit_transform(self, X, y = None, **fit_params):
        return super().fit_transform(X, y, **fit_params)
    
@dataclass
class CombinedTextCleaner(BaseEstimator, TransformerMixin, _SetOutputMixin):
    weight_title: int = 2
    weight_article: int = 1
    
    def get_feature_names_out(self, input_features=None):
        return np.array(['combined_text'])
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        t = X['title'].apply(clean_text)
        a = X['article'].apply(clean_text)
  
        return (t + " ")*self.weight_title  + (" " + a)*self.weight_article
    
    def fit_transform(self, X, y = None, **fit_params):
        return super().fit_transform(X, y, **fit_params)

@dataclass
class TopFeatureTransformer(BaseEstimator, TransformerMixin):
    
    top_n: int = 5
    feature: str = "title_suffix"
    threshold: float = 0.95
    keep_column: bool = True
    
    
    def __post_init__(self):
        self.dict_top_values_ = None
        
    def fit(self, X, y):
        print(f"Fitting with top_n={self.top_n}, threshold={self.threshold}")
        
        X_with_labels = X.copy()
        X_with_labels['label'] = y
        
        self.dict_top_values_ = top_feature_by_category(
            X_with_labels, 
            top_n=self.top_n,
            feature=self.feature,
            threshold=self.threshold
        )
        return self
    
    def transform(self, X):
        if self.dict_top_values_ is None:
            raise ValueError("Not fitted yet")
        
        return add_is_label_feature_columns(
            X, 
            self.dict_top_values_,
            feature=self.feature,
            keep_column=self.keep_column
        )


@dataclass
class TextEncoder(BaseEstimator, TransformerMixin, _SetOutputMixin):
    
    cols: Union[List[str], Tuple[str], str] = field(default_factory=lambda: ['source'])
    encoder: str = 'catboost' #target
    binary: bool = False # for binary classification, not this case but keep it for safety
    sigma: float | None = None
    drop_original: bool = True
    
    
    def __post_init__(self):
        self._encoders = {}
        self._classes = None
            
    def _clean_input(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.cols].fillna("missing")
    
    def get_feature_names_out(self, input_features=None):
        cols_list = self.cols
        names = []
        for cls in self._classes:
            for col in cols_list:
                names.append(f"{col}_prob_cl_{cls}")
        return np.array(names)
    
    

    
    def fit(self, X, y):
        if y is None:
            raise ValueError("y cannot be None for target encoding")

        self._classes = np.unique(y)
        X_cleaned = self._clean_input(X)
        
        
        for cls in self._classes:
            y_binary = (y == cls).astype(int)
            CBE = CatBoostEncoder(cols=self.cols, sigma=self.sigma, verbose = 1, return_df=True)
            CBE.fit(X_cleaned, y_binary)
            self._encoders[cls] = CBE
            
        return self
    
    
    def transform(self, X):
        X_out = X.copy()
        X_cleaned = self._clean_input(X)

        for cls, enc in self._encoders.items():
            encoded_cols = enc.transform(X_cleaned)
            encoded_cols.index = X.index
            
            new_names = {col: f"{col}_prob_cl_{cls}" for col in self.cols}
            encoded_cols = encoded_cols.rename(columns=new_names)
            X_out = pd.concat([X_out, encoded_cols], axis=1)
            
        if self.drop_original:
            X_out = X_out.drop(columns=self.cols)
            
        return X_out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None, **fit_params) -> pd.DataFrame:

        if y is None:
            raise ValueError("y cannot be None for target encoding")
            
        self._classes = np.unique(y)
        X_out = X.copy()
        X_cleaned = self._clean_input(X)
        
        for cls in self._classes:
            y_binary = (y == cls).astype(int)
            CBE = CatBoostEncoder(cols=self.cols, sigma=self.sigma, verbose = 1, return_df=True)
            

            encoded_cols = CBE.fit_transform(X_cleaned, y_binary)
            encoded_cols.index = X.index
            
            self._encoders[cls] = CBE
            
            new_names = {col: f"{col}_prob_cl_{cls}" for col in self.cols}
            encoded_cols = encoded_cols.rename(columns=new_names)
            X_out = pd.concat([X_out, encoded_cols], axis=1)
            
        if self.drop_original:
            X_out = X_out.drop(columns=self.cols)
            
        return X_out 
        
@dataclass
class CyclicDateTransformer(BaseEstimator, TransformerMixin, _SetOutputMixin):
    column: str = 'timestamp'
    drop_original: bool = True
    
    def __post_init__(self):
        self.median_timestamp_ = None
        
    def get_feature_names_out(self, input_features=None):
        c = self.column
        return np.array([
            f'is_{c}_missing', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos', 'week_sin', 'week_cos'
        ])
    
    def fit(self, X, y=None):
        ts_temp = pd.to_datetime(X[self.column], errors='coerce')
        self.median_timestamp_ = ts_temp.median()
        return self
    
    def _apply_cyclic_transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X_out = X.copy()
        
      
        ts = pd.to_datetime(X_out[self.column], errors='coerce')
        
        
        X_out[f'is_{self.column}_missing'] = ts.isna().astype(int)
        
  
        ts = ts.fillna(self.median_timestamp_)
        

        hour = ts.dt.hour
        day_of_week = ts.dt.dayofweek
        month = ts.dt.month
        week = ts.dt.isocalendar().week.astype(int)
        

        X_out['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        X_out['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
  
        X_out['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        X_out['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        

        X_out['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
        X_out['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
        

        X_out['week_sin'] = np.sin(2 * np.pi * (week - 1) / 52)
        X_out['week_cos'] = np.cos(2 * np.pi * (week - 1) / 52)
        
        if self.drop_original:
            X_out = X_out.drop(columns=[self.column])
            
        return X_out
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.median_timestamp_ is None:
            raise ValueError("Fit the transformer before calling transform.")
        return self._apply_cyclic_transform(X)

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
    

#=====================================================================================================
# PIPELINE
#=====================================================================================================`

def pipeline(
    tfidf_args
    ) -> Pipeline:
    
    # Feature Generator
    feature_generator = FeatureGenerator()
    
    
    # Text processing pipeline
    text_pipeline = Pipeline([
        ('clean_text', CombinedTextCleaner(weight_title=2, weight_article=1)),
        ('tfidf_vectorizer', TfidfVectorizer(**tfidf_args))
    ])
    
    
    # categorical features pipeline
    cat_cols = ['source','first_link_domain','title_suffix']
    text_encoder = TextEncoder(cols=cat_cols, encoder='catboost', binary=False, sigma=None, drop_original=True)
    
    # timestamp cyclic features pipeline
    timestamp_transformer = CyclicDateTransformer(column='timestamp', drop_original=True)
    
    # Numerical cols
    num_cols = ['page_rank','n_links','n_images','n_ads','n_feeds', 'article_length']
    
    processor = ColumnTransformer(
        transformers=[
            ('text_pipeline', text_pipeline, ['title', 'article']),
            ('text_encoder', text_encoder, cat_cols),
            ('timestamp_transformer', timestamp_transformer, ['timestamp']),
            ('scale_num', StandardScaler(), num_cols)
        ],
        remainder='drop'
    )
    
    full_pipeline = Pipeline([
        ('feature_generator', feature_generator),
        ('processor', processor)
    ])
    
    return full_pipeline
    