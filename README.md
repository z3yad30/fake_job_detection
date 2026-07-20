# Fake Job Detector

A Python-based fraud detection project that analyzes job posting text, engineered features, and mined red-flag patterns to predict whether a posting is fraudulent.

<!-- Build status badge placeholder -->
<!-- License badge placeholder -->
<!-- Language badge placeholder -->

## Overview

This project processes the Fake Job Postings dataset to detect fraudulent job listings using text normalization, TF-IDF feature extraction, red-flag pattern mining, and classification models. It is intended for developers and analysts who want to reproduce the data pipeline and run an interactive Streamlit app for fraud prediction.

## Features

- Streamlit UI for manual job posting input and bulk CSV upload prediction
- Text normalization and stopword removal using NLTK
- TF-IDF unigram and n-gram feature extraction for job posting text
- Red-flag pattern extraction from text and metadata (e.g. missing logo, vague salary, urgency language)
- Fraud score and red-flag count feature engineering
- Model artifact loading from serialized `models/` pickles
- CSV output and results generation for model comparison and predictions

## Tech Stack

- Python
- Streamlit
- pandas
- numpy
- NLTK
- scikit-learn
- mlxtend
- imbalanced-learn
- XGBoost
- matplotlib
- seaborn
- joblib

## Architecture

The repository is organized around a data science pipeline and a Streamlit application.

- `data/raw/`: source dataset `fake_job_postings.csv`
- `data/processed/`: cleaned data, engineered CSVs, TF-IDF feature tables, and mined pattern outputs
- `data/splits/`: train/test and scaled split CSV files used for model training and evaluation
- `models/`: serialized model and scaler artifacts loaded by the Streamlit app
- `results/`: model comparison, prediction results, and visualization outputs
- `Note_Books/`: analysis and pipeline notebooks, plus `07_streamlit.py` for the interactive app

The main runtime flow is in `Note_Books/07_streamlit.py`, which loads preprocessing artifacts and a trained model to score new postings. The notebooks implement the preprocessing, NLP pipeline, pattern mining, feature engineering, and model training steps that generate the artifacts under `data/processed/` and `models/`.

## Installation

1. Create and activate a Python virtual environment:

```powershell
cd "d:\University\semester_6\Data Mining techniques\Project\fake_job_detection"
python -m venv .venv
.\.venv\Scripts\Activate.bat
```

2. Install the required Python packages:

```powershell
pip install streamlit pandas numpy nltk scikit-learn mlxtend imbalanced-learn xgboost matplotlib seaborn joblib
```

3. Download NLTK stopwords if needed by running the Streamlit app or manually:

```powershell
python -c "import nltk; nltk.download('stopwords')"
```

## Usage

Run the Streamlit app from the repository root:

```powershell
streamlit run Note_Books/07_streamlit.py
```

Then open the local Streamlit URL in a browser. The app can:

- upload `fake_job_postings.csv` for batch prediction
- accept manual job posting fields
- display a fraud prediction, probability, and detected red flags

If you want to inspect the analysis pipeline, open the notebooks in `Note_Books/` with Jupyter:

```powershell
jupyter notebook Note_Books/03_nlp_pipeline.ipynb
```

## Project Structure

- `Note_Books/07_streamlit.py` — Streamlit application for fraud detection and interactive prediction
- `Note_Books/03_nlp_pipeline.ipynb` — NLP preprocessing, normalization, tokenization, TF-IDF vectorization
- `Note_Books/04_pattern_mining.ipynb` — association rule mining and red-flag pattern creation
- `Note_Books/05_feature_engineering.ipynb` — feature construction, encoding, and pipeline bridging
- `Note_Books/06_model_training.ipynb` — training and evaluating classification models
- `data/raw/fake_job_postings.csv` — original job posting dataset
- `data/processed/cleaned_data.csv` — cleaned dataset used for feature engineering
- `data/processed/vectorized_features.csv` — TF-IDF feature table
- `data/processed/frequency_table_bigrams.csv` — extracted bigram frequencies
- `data/processed/frequency_table_trigrams.csv` — extracted trigram frequencies
- `data/processed/mined_patterns.csv` — mined fraud pattern summaries
- `data/processed/mined_rules.csv` — association rules output
- `data/splits/` — training and test split CSV files
- `models/` — serialized scaler and model `.pkl` artifacts loaded by the app
- `results/` — evaluation outputs, model comparison, and prediction results
- `Fake_Job_Postings_Project_Plan.md` — project plan and methodology notes

## License

No license specified. Replace this section with an appropriate license if one is added.
