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

        logger.debug('TF-IDF Applied for train dataset with size %s',X_train.shape)

        save_model(vectorizer, os.path.join(get_root_directory(), 'saved_model/tfidf_vectorizer.pkl'))

        logger.debug('TF-IDF applied with trigrams and data transformed')
        return X_train, y_train
    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise

def get_root_directory() -> str:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.join(current_directory, '../../'))

def train_model(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:
    try:
        best_model = lgb.LGBMClassifier(
            objective='binary',
            num_class=2,
            metric='binary_logloss',
            is_unbalance=True,
            class_weight='balanced',
            reg_alpha=0.1,
            reg_lambda=0.1,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        best_model.fit(X_train, y_train)
        logger.debug('LightLGBM model trained with learning_rate=%f, max_depth=%d, n_estimators=%d', learning_rate, max_depth, n_estimators)

        return best_model
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        root_directory = get_root_directory()
        os.makedirs("saved_model", exist_ok=True)
        params = load_params(os.path.join(root_directory,'params.yaml'))
        
        data_path = params['model_building']['train_data_path']
        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])
        n_estimators = params['model_building']['n_estimators']
        max_depth = params['model_building']['max_depth']
        learning_rate = params['model_building']['learning_rate']

        train_data = load_data(os.path.join(root_directory,data_path))

        X_train, y_train = apply_tfidf(train_data, max_features, ngram_range)
        model = train_model(
            X_train=X_train,
            y_train=y_train,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        save_model(model, os.path.join(root_directory,'saved_model/lgbm_model.pkl'))

    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()