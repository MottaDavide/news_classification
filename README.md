# News Classification

A machine learning pipeline for news article classification using TF-IDF features, metadata extraction, and ensemble models (LinearSVC / LightGBM).

## Project Structure

```
news_classification/
├── config/
│   └── config.yaml        # Configuration file
├── data/
│   ├── development.csv    # Training data
│   └── evaluation.csv     # Test data for submission
├── models/                # Saved models
├── utils/
│   ├── preproc.py         # Data preprocessing
│   ├── feature_extrc.py   # Feature extraction
│   └── train_eval.py      # Training and evaluation
├── main.py                # Main entry point
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

### 3. Install category_encoders via Conda (if pip fails)

If you encounter issues installing `category_encoders` (which provides `CatBoostEncoder`) via pip, you can use conda:

```bash
# First, deactivate the virtual environment if active
deactivate

# Create a conda environment instead
conda create -n news_classification python=3.11
conda activate news_classification

# Install category_encoders via conda-forge
conda install -c conda-forge category_encoders

# Install remaining dependencies via pip
pip install -r requirements.txt
```

Alternatively, if you want to keep your venv and just install category_encoders separately:

```bash
# Try installing from conda-forge using pip
pip install --no-cache-dir category_encoders

# Or install specific version
pip install category_encoders==2.9.0
```

## Running the Pipeline

### Basic Usage

```bash
python main.py
```

This will:
1. Load data from `data/development.csv` and `data/evaluation.csv`
2. Train and evaluate a model using the settings in `config/config.yaml`
3. Display evaluation metrics and confusion matrix

## Configuration

All configuration options are defined in `config/config.yaml`:

### Main Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `RANDOM_STATE` | int | `42` | Random seed for reproducibility |
| `CONTEXT_INJECTION` | bool | `True` | Inject metadata (source, domain) into text features |
| `USE_CATBOOST` | bool | `True` | Use CatBoost encoding for categorical features |
| `TUNING` | bool | `False` | Enable hyperparameter tuning via GridSearch/RandomSearch |

### Label Configuration

```yaml
LABEL_NAMES:
  0: 'International News'
  1: 'Business'
  2: 'Technology'
  3: 'Entertainment'
  4: 'Sports'
  5: 'General News'
  6: 'Health'
```

### TF-IDF Parameters

```yaml
TFIDF_PARMAS:
  max_features: 15000      # Maximum vocabulary size
  ngram_range: (1, 3)      # Use unigrams, bigrams, and trigrams
  min_df: 5                # Minimum document frequency
  max_df: 0.7              # Maximum document frequency
  token_pattern: r'(?u)\b[a-zA-Z]{3,}\b'  # Only words with 3+ letters
  stop_words: 'english'    # Remove English stop words
  sublinear_tf: True       # Apply sublinear TF scaling
```

### CatBoost Encoder Parameters

```yaml
CATBOOST_PARAMS:
  sigma: 0.9               # Smoothing parameter for CatBoost encoding
```

## Model Types

The pipeline supports two model types (configured in `main.py`):

### LinearSVC (`model_type='svc'`)
- Fast linear Support Vector Classifier
- Good for high-dimensional sparse TF-IDF features
- Default hyperparameters when tuning is disabled:
  - `C=0.05`
  - `loss='squared_hinge'`
  - `class_weight='balanced'`

### LightGBM (`model_type='lgbm'`)
- Gradient boosting model
- Better for mixed feature types (text + numerical + categorical)
- Default hyperparameters when tuning is disabled:
  - `n_estimators=500`
  - `max_depth=7`
  - `learning_rate=0.05`
  - `num_leaves=31`

## Configuration Examples

### Example 1: Quick training without tuning

Edit `config/config.yaml`:
```yaml
TUNING: False
CONTEXT_INJECTION: True
USE_CATBOOST: False
```

### Example 2: Full tuning with CatBoost encoding

Edit `config/config.yaml`:
```yaml
TUNING: True
CONTEXT_INJECTION: True
USE_CATBOOST: True
```

### Example 3: Use LightGBM model

Edit `main.py` line 65:
```python
model_type='lgbm',
```

## Generating Submission

To generate predictions for the evaluation set, uncomment the submission block in `main.py`:

```python
# Generate submission
submission = train_full_and_predict(
    dev_df,
    eval_df,
    model_type='svc',
    model=model,
    pipeline=None,
    context_injection=CONTEXT_INJECTION,
    use_catboost=USE_CATBOOST,
    submission_path='submission.csv'
)
```

This will create a `submission.csv` file with predictions.

## Report
The `report` folder contains the technical report. Please open `main.pdf`.