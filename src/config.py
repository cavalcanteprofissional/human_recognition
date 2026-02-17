import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Diretórios raiz
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# Criar diretórios se não existirem
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configurações do Kaggle
KAGGLE_CONFIG = {
    "username": os.getenv("KAGGLE_USERNAME"),
    "key": os.getenv("KAGGLE_KEY")
}

# Configurações do dataset
DATASET_NAME = "constantinwerner/human-detection-dataset"
DATASET_PATH = RAW_DATA_DIR / "human-detection-dataset"

# Configurações de processamento
TARGET_SIZE = (256, 256)
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Configurações do LBP
LBP_CONFIG = {
    "radius": 1,
    "n_points": 8 * 1,  # 8 * radius
    "method": "uniform"
}

# Configurações de treinamento
RF_CONFIG = {
    "n_estimators": [10, 50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}