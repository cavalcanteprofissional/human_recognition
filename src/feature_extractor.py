import numpy as np
from skimage import io, color
from skimage.feature import local_binary_pattern
from skimage.transform import resize
from typing import List, Union, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

from src.config import LBP_CONFIG, TARGET_SIZE

logger = logging.getLogger(__name__)

class LBPFeatureExtractor:
    """
    Extrator de características usando Local Binary Patterns (LBP).
    Implementação detalhada para explicação no relatório.
    """
    
    def __init__(self, radius: int = 1, n_points: int = 8, method: str = 'uniform'):
        """
        Inicializa o extrator LBP.
        
        Args:
            radius: Raio do círculo de vizinhança
            n_points: Número de pontos na vizinhança circular
            method: Método de quantização ('uniform', 'ror', 'default')
        """
        self.radius = radius
        self.n_points = n_points
        self.method = method
        
        # Para método uniform, o número de bins é n_points + 2
        self.n_bins = n_points + 2 if method == 'uniform' else 256
        
        logger.info(f"LBP Extractor inicializado: radius={radius}, n_points={n_points}, method={method}")
    
    def _calculate_lbp(self, image: np.ndarray) -> np.ndarray:
        """
        Calcula a imagem LBP.
        
        Explicação detalhada para o relatório:
        O LBP compara cada pixel com seus vizinhos em um círculo de raio R.
        Para cada vizinho:
        1. Se o valor do vizinho >= pixel central, atribui 1
        2. Caso contrário, atribui 0
        Isso forma um número binário que descreve a textura local.
        
        Args:
            image: Imagem em escala de cinza
            
        Returns:
            Imagem LBP
        """
        return local_binary_pattern(
            image, 
            self.n_points, 
            self.radius, 
            self.method
        )
    
    def _calculate_histogram(self, lbp_image: np.ndarray) -> np.ndarray:
        """
        Calcula o histograma normalizado da imagem LBP.
        
        O histograma das ocorrências de cada padrão LBP forma um
        vetor de características que descreve a textura global da imagem.
        
        Args:
            lbp_image: Imagem LBP
            
        Returns:
            Histograma normalizado
        """
        hist, _ = np.histogram(
            lbp_image.ravel(),
            bins=np.arange(0, self.n_bins + 1),
            range=(0, self.n_bins)
        )
        
        # Normalização L1
        hist = hist.astype("float")
        hist_norm = hist / (hist.sum() + 1e-6)
        
        return hist_norm
    
    def extract_from_path(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Extrai características LBP de uma imagem a partir do caminho.
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Vetor de características LBP
        """
        try:
            # Carregar imagem
            image = io.imread(str(image_path))
            
            # Converter para escala de cinza se necessário
            if len(image.shape) == 3:
                image = color.rgb2gray(image)
            
            # Redimensionar se necessário
            if image.shape[:2] != TARGET_SIZE:
                image = resize(image, TARGET_SIZE, preserve_range=True)
            
            # Extrair características
            return self.extract_from_image(image)
            
        except Exception as e:
            logger.error(f"Erro ao extrair características de {image_path}: {e}")
            return None
    
    def extract_from_image(self, image: np.ndarray) -> np.ndarray:
        """
        Extrai características LBP de uma imagem já carregada.
        
        Args:
            image: Imagem em escala de cinza
            
        Returns:
            Vetor de características LBP
        """
        # Calcular LBP
        lbp = self._calculate_lbp(image)
        
        # Calcular histograma
        hist = self._calculate_histogram(lbp)
        
        return hist
    
    def _extract_single(self, img_path: Union[str, Path]) -> Tuple[Union[np.ndarray, None], int]:
        """
        Extrai características de uma única imagem (para paralelização).
        
        Args:
            img_path: Caminho para a imagem
            
        Returns:
            Tupla (vetor de características, índice original)
        """
        try:
            image = io.imread(str(img_path))
            
            if len(image.shape) == 3:
                image = color.rgb2gray(image)
            
            if image.shape[:2] != TARGET_SIZE:
                image = resize(image, TARGET_SIZE, preserve_range=True)
            
            lbp = self._calculate_lbp(image)
            hist = self._calculate_histogram(lbp)
            return hist
            
        except Exception as e:
            logger.error(f"Erro ao extrair características de {img_path}: {e}")
            return None
    
    def extract_batch(self, image_paths: List[Union[str, Path]], 
                     show_progress: bool = True,
                     n_jobs: int = -1) -> Tuple[np.ndarray, List[int]]:
        """
        Extrai características de um lote de imagens com paralelização.
        
        Args:
            image_paths: Lista de caminhos de imagens
            show_progress: Mostrar barra de progresso
            n_jobs: Número de jobs paralelos (-1 = todos os cores)
            
        Returns:
            Tupla (matriz de características, índices válidos)
        """
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        if n_jobs == 1:
            return self._extract_batch_sequential(image_paths, show_progress)
        
        logger.info(f"Extraindo características em paralelo com {n_jobs} workers...")
        
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(self._extract_single)(img_path) 
            for img_path in tqdm(image_paths, desc="Extraindo características")
        )
        
        features = []
        valid_indices = []
        
        for i, feat in enumerate(results):
            if feat is not None:
                features.append(feat)
                valid_indices.append(i)
        
        if not features:
            raise ValueError("Nenhuma característica válida extraída")
        
        logger.info(f"Extração concluída: {len(features)} imagens processadas")
        
        return np.array(features), valid_indices
    
    def _extract_batch_sequential(self, image_paths: List[Union[str, Path]], 
                                  show_progress: bool = True) -> Tuple[np.ndarray, List[int]]:
        """
        Versão sequencial da extração (fallback).
        """
        features = []
        valid_indices = []
        
        iterator = tqdm(image_paths, desc="Extraindo características") if show_progress else image_paths
        
        for i, img_path in enumerate(iterator):
            feat = self._extract_single(img_path)
            if feat is not None:
                features.append(feat)
                valid_indices.append(i)
        
        if not features:
            raise ValueError("Nenhuma característica válida extraída")
        
        return np.array(features), valid_indices
    
    def explain_lbp(self) -> str:
        """
        Retorna uma explicação detalhada do LBP para o relatório.
        """
        explanation = """
        LOCAL BINARY PATTERNS (LBP) - EXPLICAÇÃO DETALHADA
        
        O LBP é um descritor de textura local que se tornou popular devido à sua
        eficiência computacional e robustez a variações de iluminação.
        
        PRINCÍPIO DE FUNCIONAMENTO:
        
        1. Para cada pixel da imagem, o LBP compara seu valor com os valores dos
           pixels vizinhos dispostos em um círculo de raio R.
        
        2. Se o valor do vizinho for maior ou igual ao pixel central, atribui-se 1;
           caso contrário, atribui-se 0.
        
        3. Isso gera um número binário de N bits (onde N é o número de vizinhos).
        
        4. Este número binário é convertido para decimal, resultando em um rótulo
           para o pixel central.
        
        EXEMPLO:
        Vizinhança 3x3 (R=1, N=8):
        
        [85  32  26]      Vizinhos: 85, 32, 26, 61, 9, 78, 12, 65
        [61  50  9]   ->  Pixel central: 50
        [78  12  65]      Threshold: 1 se >=50, 0 se <50
                          Resultado binário: 11010010 (decimal: 210)
        
        CARACTERÍSTICAS UNIFORMES:
        Para reduzir a dimensionalidade, usa-se a versão "uniforme" onde apenas
        padrões com no máximo 2 transições 0-1 são considerados. Isso reduz de
        256 para 59 padrões possíveis.
        
        VANTAGENS:
        - Invariância a mudanças monotônicas de iluminação
        - Baixo custo computacional
        - Boa capacidade de descrição de texturas
        
        DESVANTAGENS:
        - Sensível a rotações (versão circular resolve parcialmente)
        - Pode perder informações em escalas muito diferentes
        """
        return explanation