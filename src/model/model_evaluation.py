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
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

load_dotenv(override=True)

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

def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.join(current_dir, '../../'))

def load_params(params_path: str) -> dict:
    try:
        logger.debug('Attempting to load parameters from %s', params_path)
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('Parameters file not found at %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Error parsing YAML file at %s: %s', params_path, e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading parameters: %s', e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    try:
        logger.debug('Attempting to load data from %s', data_url)
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
        logger.debug('Attempting to load model from %s', model_path)

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
        logger.debug('Attempting to load TF-IDF vectorizer from %s', vectorizer_path)

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

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    try:
        logger.debug("Starting model evaluation")

        y_pred = model.predict(X_test)
        report = classification_report(y_test,y_pred,output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        logger.debug("Model evaluation completed successfully")

        return report, cm
    
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def log_confusion_matrix(cm, dataset_name):
    try:
        logger.debug('Logging confusion matrix')

        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d',cmap='Blues')
        plt.title(f'Confusion Matrix for {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        cm_file_path = f"Confusion_matrix_{dataset_name}.png"
        plt.savefig(cm_file_path)
        mlf.log_artifact(cm_file_path)
        logger.debug('Confussion matrix logged successfully')
    
    except Exception as e:
        logger.error('Error logging confussion matrix: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        logger.debug("Creating model info")

        model_info={
            "run_id": run_id,
            "model_path": model_path
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    MLFLOW_URL = os.getenv("MLFLOW_ENDPOINT_URL")
    mlf.set_tracking_uri(MLFLOW_URL)
    mlf.set_experiment("IMDB_Sentiment_Analysis using DVC pipeline runs")

    with mlf.start_run() as run:
        try:
            root_dir = get_root_directory()
            params_path = os.path.join(root_dir, 'params.yaml')
            params = load_params(params_path=params_path)

            for key, value in params.items():
                mlflow.log_param(key, value)
            
            model = load_model(os.path.join(root_dir, 'saved_model/lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'saved_model/tfidf_vectorizer.pkl'))

            test_data_path = params['model_evaluation']['test_data_path']

            x_text = vectorizer.transform(test_data_path['clean_text'])
            x_num = test_data_path.drop(columns=['clean_text','review', 'sentiment'])
            X_test = np.hstack([x_text.toarray(), x_num.values])
            y_test = test_data_path['sentiment'].values

            input_example = pd.DataFrame(X_test.to_array()[:5], columns=vectorizer.get_feature_names_out())
            signature = infer_signature(input_example, model.predict(X_test[:5]))

            mlflow.sklearn.log_model(
                model,
                artifact_path='model',
                signature=signature,
                input_example=input_example
            )

            model_path = 'model'
            save_model_info(
                run_id=run.info.run_id,
                model_path=model_path,
                file_path='experiment_model_info.json'
            )

            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))
            report, cm = evaluate_model(model, X_test, y_test)

            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            log_confusion_matrix(cm, "Test Data")

            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")

if __name__ == '__main__':
    main()