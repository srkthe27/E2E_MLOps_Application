import os
import yaml
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.custom_preprocess_helper import CustomNLPDatasetOp,CustomNLPPreprocessor

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

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

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str, preprocessed: bool = False) -> None:
    try:
        if not preprocessed:
            raw_data_path = os.path.join(data_path,'raw')
            os.makedirs(raw_data_path, exist_ok=True)
            train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
            test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
            
            logger.debug('Train and test data saved to %s', raw_data_path)

        else:
            preprocessed_data_path = os.path.join(data_path,'processed')
            os.makedirs(preprocessed_data_path, exist_ok=True)
            train_data.to_csv(os.path.join(preprocessed_data_path, "preprocessed_train.csv"), index=False)
            test_data.to_csv(os.path.join(preprocessed_data_path, "preprocessed_test.csv"), index=False)

            logger.debug('Preprocessed data saved to %s', preprocessed_data_path)

    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        preprocessor = CustomNLPPreprocessor()
        dataset_op = CustomNLPDatasetOp(
            df=df,
            text_col='review',
            target_col='sentiment',
            preprocessor=preprocessor
        )
        new_df = dataset_op.run_dataset_operations(verbose=True)
        logger.debug('Data preprocessing completed.')
        return new_df
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def main():
    try:
        logger.debug('Starting data ingestion and preprocessing...')

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
        params = load_params(os.path.join(PROJECT_ROOT, "params.yaml"))
        relative_data_path = params['data_ingestion']['data_path']
        DATA_URL = os.path.join(PROJECT_ROOT, relative_data_path)
        test_size = params['data_ingestion']['test_size']

        df = load_data(data_url=DATA_URL)

        train_data, test_data = train_test_split(df,test_size=test_size,random_state=42)
        logger.debug('Raw data splited into train and test sets with test size %s', test_size)

        save_data(train_data,test_data,data_path=os.path.join(PROJECT_ROOT,'data'))
        logger.debug('Raw data saved successfully')

        processed_train_data = preprocess_data(train_data)
        processed_test_data = preprocess_data(test_data)
        logger.debug('Data preprocessing completed successfully')

        save_data(processed_train_data, processed_test_data, data_path=os.path.join(PROJECT_ROOT,'data'), preprocessed=True)

    except Exception as e:
        logger.exception('Failed to complete the data ingestion and preprocessing process: %s', e)

if __name__ == '__main__':
    main()