import os
import yaml
import json
import pickle
import logging
import mlflow.sklearn
import numpy as np
import pandas as pd
import mlflow as mlf
import seaborn as sns
import matplotlib.pyplot as plt
from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('Model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

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

def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except FileNotFoundError:
        logger.error('Model file not found at %s', model_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except FileNotFoundError:
        logger.error('Vectorizer file not found at %s', vectorizer_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the vectorizer: %s', e)
        raise

