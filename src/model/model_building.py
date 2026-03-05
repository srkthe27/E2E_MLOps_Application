import os
import yaml
import pickle
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('Model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameter retrived from %s',params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    try:
        _, ext = os.path.splitext(data_url)
        ext = ext.lower()

        readers = {
            ".csv": pd.read_csv,
            ".xlsx": pd.read_excel,
            ".xls": pd.read_excel,
            ".json": pd.read_json,
            ".parquet": pd.read_parquet
        }

        if ext not in readers:
            raise ValueError(f"Unsupported file format: {ext}")

        df = readers[ext](data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        x_text = vectorizer.fit_transform(train_data['review'])
        x_num = train_data.drop(columns=['review', 'sentiment'])
        X_train = np.hstack([x_text.toarray(), x_num.values])
        y_train = train_data['sentiment'].values

        logger.debug('TF-IDF Applied for train dataset with size %d',X_train.shape)

        with open(os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.debug('TF-IDF applied with trigrams and data transformed')
        return X_train, y_train
    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise

def get_root_directory() -> str:
    root_directory = os.path.dirname(os.path.abspath(__file__))
    return root_directory

