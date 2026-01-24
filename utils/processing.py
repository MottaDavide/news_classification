import pandas as pd
from pathlib import Path
import re
import numpy as np
import yaml
from urllib.parse import urlparse
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin # I need BaseEstimators cause it has get_params and set_params methods useful for tuning
from sklearn.pipeline import Pipeline
from dataclasses import dataclass



with open("config/config.yaml", 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)
    
LABEL_NAMES = config['LABEL_NAMES'][0]


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
    
    
    temp_df = df.copy()
    
    # Converting timestamp to datetime
    print("Converting timestamp to datetime...")
    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], errors='coerce')
    
    # Dropping duplicates based on source, title, article, label keeping the most recent one
    print("Dropping duplicates based on source, title, article, label keeping the most recent one...")
    temp_df = temp_df.sort_values(by=['source', 'title', 'article', 'label', 'timestamp'], ascending=[True, True, True, True, False])
    temp_df = temp_df.drop_duplicates(subset = ['source','title','article','label'])
    
    # Dropping duplicates that have the same source, title, article but different label. This is to avoid ambiguity
    print("Dropping duplicates that have the same source, title, article but different label...")
    grouped = temp_df.groupby(by = ['source','title','article'], as_index=False)['label'].agg({'nunique'})
    index = grouped[grouped['nunique'] >= 2].sort_values(by='nunique', ascending=False).index
    temp_df = temp_df.iloc[~temp_df.index.isin(index)]
    
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
    5. Removes URLs (http, https, www).
    6. Removes special characters, keeping only alphanumeric characters and spaces.
    7. Removes extra whitespace (converts multiple spaces to a single space).
    
    This preprocessing is intended to clean the text for tasks like text mining or machine learning.
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
    
    # 8. Remove extra whitespace (convert multiple spaces into a single space)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


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

#=====================================================================================================
# TITLE SUFFIX EXTRACTION
#=====================================================================================================


def extract_title_suffix_features(title: str) -> str | None:
    """
    Some titles have suffixes in brackets that provide additional context.
    
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

def apply_title_extraction(df: pd.DataFrame) -> pd.DataFrame:
    
    temp_df = df.copy()
    
    temp_df['title_suffix'] = temp_df['title'].apply(extract_title_suffix_features)
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
    else:
        raise ValueError("Feature must be one of ['title_suffix', 'first_link_domain']")
    
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
    
    

    
    for key, value in dict_top_values.items():
        col_name = f'is_{str(key).lower()}_{feature}'
        df[col_name] = df[feature].isin(value).astype(int)
        
    if not keep_column:
        df.drop(columns=[feature], inplace=True)
    
    return df
    
    

#=====================================================================================================
# PIPELINE
#=====================================================================================================`

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
    

        
    
    
    
    
    


