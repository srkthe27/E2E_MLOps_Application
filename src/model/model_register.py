import os, json, logging, mlflow
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    try:
        logger.debug("Attempting to load model info from %s", file_path)
        with open(file_path, 'r') as f:
            model_info = json.load(f)
        logger.debug("Model info successfully loaded from %s", file_path)
        return model_info
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except json.JSONDecodeError as e:
        logger.error("JSON decode error in file %s: %s", file_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading model info from %s: %s", file_path, e)
        raise

def model_register(model_name: str, model_info: dict):
    try:
        logger.debug("Starting model registration for %s", model_name)

        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def get_root_directory():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.join(current_dir, '../../'))

def main():
    mlflow.set_tracking_uri(os.getenv('MLFLOW_ENDPOINT_URL'))
    try:
        root_dir = get_root_directory()
        model_info_path = os.path.join(root_dir, 'experiment_model_info.json')
        model_info = load_model_info(model_info_path)
        model_register(model_name='LGBM model',model_info=model_info)
    
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()