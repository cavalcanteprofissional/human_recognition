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

# Divisão de dados (Train/Val/Test)
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# Validação cruzada
CV_FOLDS = 5
CV_SCORING = "accuracy"

# Configurações de modelos avançados
ADVANCED_MODELS_CONFIG = {
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "gradient_boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 7]
    },
    "xgboost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 7]
    },
    "lightgbm": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 7, -1]
    },
    "svm": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"]
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    },
    "logistic_regression": {
        "C": [0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [1000]
    },
    "mlp": {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001],
        "max_iter": [500]
    }
}

# Modelos para ensemble (serão treinados individualmente primeiro)
ENSEMBLE_BASE_MODELS = ["random_forest", "gradient_boosting", "xgboost", "svm", "logistic_regression"]

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

# ============================================
# CONFIGURAÇÕES DA CÂMERA YOOSEE
# ============================================

# Configurações da câmera Yoosee (carregadas do .env)
YOOSEE_CONFIG = {
    "ip": os.getenv("YOOSEE_IP", "192.168.1.100"),  # IP da câmera
    "port": int(os.getenv("YOOSEE_PORT", "554")),   # Porta RTSP (padrão 554)
    "username": os.getenv("YOOSEE_USERNAME", "admin"),  # Usuário
    "password": os.getenv("YOOSEE_PASSWORD", ""),   # Senha
    "stream": os.getenv("YOOSEE_STREAM", "onvif1")  # onvif1 (principal) ou onvif2 (sub)
}

# URLs RTSP pré-configuradas para diferentes modelos Yoosee [citation:3][citation:5]
YOOSEE_RTSP_PATHS = {
    "onvif1": "/onvif1",           # Mais comum [citation:1]
    "onvif2": "/onvif2",           # Sub-stream
    "live": "/live.sdp",            # Para modelos J1080P
    "stream11": "/11",              # Para alguns modelos
    "h264": "/h264",                 # Stream H.264
    "user_pass": "/user=[USERNAME]&password=[PASSWORD]&channel=1&stream=0.sdp?"  # Formato alternativo
}

def get_yoosee_rtsp_url(stream_type="onvif1", custom_path=None, use_tcp=True):
    """
    Gera a URL RTSP para a câmera Yoosee.
    
    Args:
        stream_type: Tipo de stream ('onvif1', 'onvif2', 'live', 'stream11', 'h264')
        custom_path: Caminho personalizado (sobrescreve stream_type)
        use_tcp: Se True, força transporte TCP
    
    Returns:
        URL RTSP completa
    """
    config = YOOSEE_CONFIG
    
    # Escolher o caminho
    if custom_path:
        path = custom_path
    else:
        path = YOOSEE_RTSP_PATHS.get(stream_type, "/onvif1")
    
    # Adicionar ?tcp para forçar transporte TCP (resolve problema com algumas câmeras)
    transport = "?tcp" if use_tcp else ""
    
    # Formatar com usuário e senha se fornecidos
    if config["username"] and config["password"]:
        # URL com autenticação [citation:4][citation:8]
        url = f"rtsp://{config['username']}:{config['password']}@{config['ip']}:{config['port']}{path}{transport}"
    else:
        # URL sem autenticação
        url = f"rtsp://{config['ip']}:{config['port']}{path}{transport}"
    
    return url

# Cache da última URL gerada
YOOSEE_RTSP_URL = get_yoosee_rtsp_url(YOOSEE_CONFIG["stream"])


def reload_yoosee_config():
    """
    Recarrega as configurações da câmera Yoosee do arquivo .env.
    Útil após o .env ser atualizado dinamicamente.
    """
    global YOOSEE_CONFIG, YOOSEE_RTSP_URL
    
    YOOSEE_CONFIG = {
        "ip": os.getenv("YOOSEE_IP", "192.168.1.100"),
        "port": int(os.getenv("YOOSEE_PORT", "554")),
        "username": os.getenv("YOOSEE_USERNAME", "admin"),
        "password": os.getenv("YOOSEE_PASSWORD", ""),
        "stream": os.getenv("YOOSEE_STREAM", "onvif1")
    }
    
    YOOSEE_RTSP_URL = get_yoosee_rtsp_url(YOOSEE_CONFIG["stream"])
    
    return YOOSEE_CONFIG


def find_and_update_yoosee_ip():
    """
    Encontra a câmera Yoosee na rede e atualiza o arquivo .env.
    
    Returns:
        Tuple (ip, port, stream_type) se encontrado, (None, None, None) caso contrário
    """
    from tools.find_yoosee_ip import update_env_with_camera_ip
    
    result = update_env_with_camera_ip()
    
    if result[0] is not None:
        reload_yoosee_config()
    
    return result