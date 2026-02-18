import os
import zipfile
import shutil
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
from sklearn.model_selection import train_test_split
import kaggle
from PIL import Image
import logging
from tqdm import tqdm

from src.config import (
    KAGGLE_CONFIG, DATASET_NAME, DATASET_PATH,
    RAW_DATA_DIR, PROCESSED_DATA_DIR, TARGET_SIZE,
    RANDOM_SEED, TEST_SIZE, VALIDATION_SIZE
)

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HumanDatasetLoader:
    """Carrega e prepara o dataset de detecção humana do Kaggle."""
    
    def __init__(self):
        self.raw_path = DATASET_PATH
        self.processed_path = PROCESSED_DATA_DIR / "human_dataset"
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # Configurar credenciais do Kaggle
        os.environ['KAGGLE_USERNAME'] = KAGGLE_CONFIG["username"]
        os.environ['KAGGLE_KEY'] = KAGGLE_CONFIG["key"]
    
    def download_dataset(self, force_download: bool = False) -> Path:
        """
        Baixa o dataset do Kaggle se ele não existir.
        
        Args:
            force_download: Se True, baixa novamente mesmo se já existir
            
        Returns:
            Caminho para o dataset baixado
        """
        if self.raw_path.exists() and not force_download:
            logger.info(f"Dataset já existe em {self.raw_path}")
            return self.raw_path
        
        logger.info(f"Baixando dataset {DATASET_NAME}...")
        
        # Criar diretório temporário
        temp_dir = RAW_DATA_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Baixar dataset
            kaggle.api.dataset_download_files(
                DATASET_NAME,
                path=str(temp_dir),
                unzip=True
            )
            
            # Mover para o local final
            if self.raw_path.exists():
                shutil.rmtree(self.raw_path)
            
            # Listar o diretório para encontrar o nome correto
            downloaded_items = list(temp_dir.iterdir())
            if downloaded_items:
                downloaded_path = downloaded_items[0]
                shutil.move(str(downloaded_path), str(self.raw_path))
            
            # Limpar
            shutil.rmtree(temp_dir)
            
            logger.info(f"Dataset baixado e extraído em {self.raw_path}")
            
        except Exception as e:
            logger.error(f"Erro ao baixar dataset: {e}")
            raise
        
        return self.raw_path
    
    def organize_dataset(self) -> Dict[str, Path]:
        """
        Organiza o dataset em estrutura de pastas por classe.
        
        Returns:
            Dicionário com caminhos para as pastas de cada classe
        """
        logger.info("Organizando dataset...")
        
        # Estrutura de diretórios
        class_dirs = {
            "0": self.processed_path / "no_human",
            "1": self.processed_path / "human"
        }
        
        # Criar diretórios de classe
        for dir_path in class_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Mapear arquivos para classes (baseado no dataset original)
        # O dataset tem pastas '0' (sem humano) e '1' (com humano)
        class_0_dir = self.raw_path / "0"
        class_1_dir = self.raw_path / "1"
        
        # Processar imagens da classe 0 (sem humano)
        if class_0_dir.exists():
            for img_path in tqdm(list(class_0_dir.glob("*.png")), desc="Processando classe 0"):
                dest_path = class_dirs["0"] / img_path.name
                self._copy_and_resize_image(img_path, dest_path)
        
        # Processar imagens da classe 1 (com humano)
        if class_1_dir.exists():
            for img_path in tqdm(list(class_1_dir.glob("*.png")), desc="Processando classe 1"):
                dest_path = class_dirs["1"] / img_path.name
                self._copy_and_resize_image(img_path, dest_path)
        
        logger.info(f"Dataset organizado: {self.processed_path}")
        return class_dirs
    
    def _copy_and_resize_image(self, src_path: Path, dest_path: Path):
        """Copia e redimensiona uma imagem."""
        try:
            with Image.open(src_path) as img:
                # Converter para RGB se necessário
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Redimensionar
                img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                
                # Salvar
                img_resized.save(dest_path)
        except Exception as e:
            logger.warning(f"Erro ao processar {src_path}: {e}")
    
    def create_train_val_test_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Cria a divisão treino/validação/teste.
        
        Returns:
            Tupla com (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Criando divisão treino/validação/teste...")
        
        # Coletar todos os caminhos de imagem e rótulos
        image_paths = []
        labels = []
        
        for label in [0, 1]:
            class_dir = self.processed_path / ("no_human" if label == 0 else "human")
            for img_path in class_dir.glob("*.png"):
                image_paths.append(str(img_path))
                labels.append(label)
        
        # Converter para arrays
        X = np.array(image_paths)
        y = np.array(labels)
        
        # Primeira divisão: treino (70%) e temporário (30%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=(TEST_SIZE + VALIDATION_SIZE),
            random_state=RANDOM_SEED,
            stratify=y
        )
        
        # Segunda divisão: validação (10%) e teste (20%) a partir do temporário
        val_size = VALIDATION_SIZE / (TEST_SIZE + VALIDATION_SIZE)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=TEST_SIZE/(TEST_SIZE + VALIDATION_SIZE),
            random_state=RANDOM_SEED,
            stratify=y_temp
        )
        
        logger.info(f"Divisão criada:")
        logger.info(f"  Treino: {len(X_train)} imagens")
        logger.info(f"  Validação: {len(X_val)} imagens")
        logger.info(f"  Teste: {len(X_test)} imagens")
        
        # Salvar os splits
        np.savez(
            self.processed_path / "splits.npz",
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def load_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Carrega os dados processados.
        
        Returns:
            Dicionário com os splits de dados
        """
        splits_path = self.processed_path / "splits.npz"
        
        if not splits_path.exists():
            logger.info("Arquivo de splits não encontrado. Processando dados...")
            self.download_dataset()
            self.organize_dataset()
            self.create_train_val_test_split()
        
        # Carregar splits
        data = np.load(splits_path)
        
        return {
            "train": (data['X_train'], data['y_train']),
            "val": (data['X_val'], data['y_val']),
            "test": (data['X_test'], data['y_test'])
        }

# Exemplo de uso
if __name__ == "__main__":
    loader = HumanDatasetLoader()
    data = loader.load_data()
    
    print("\nResumo dos dados:")
    for split_name, (X, y) in data.items():
        print(f"{split_name}: {len(X)} imagens")
        print(f"  Classe 0 (sem humano): {sum(y == 0)}")
        print(f"  Classe 1 (com humano): {sum(y == 1)}")