import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
import re

import yaml as yml


with open('config/config.yaml', 'r') as f:
    config = yml.safe_load(f)
    


LABEL_NAMES = config['LABEL_NAMES']

RSS_TO_LABEL = config['RSS_TO_LABEL']

RANDOM_STATE = config['RANDOM_STATE']

CONTEXT_INJECTION= config['CONTEXT_INJECTION']

USE_CATBOOST= config['USE_CATBOOST']

warnings.filterwarnings('ignore')

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
    
    # Remove naive words
    text = re.sub(r'\b(said|says|say|year|years|for|and|the|you|day|not|but|she|he|his|her|your|yours|they|them|their)\b', '', text)
    
    # Remove standalone numbers
    text = re.sub(r'\b\d+\b', '', text)
    
    return text

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