import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
import json

from src.config import MODELS_DIR, REPORTS_DIR, RF_CONFIG, RANDOM_SEED
from src.feature_extractor import LBPFeatureExtractor
from src.data_loader import HumanDatasetLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Treina e avalia modelos de classificação para detecção de humanos."""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.reports_dir = REPORTS_DIR
        self.results = {}
        
    def extract_features_from_data(self, X_paths: np.ndarray, 
                                   extractor: LBPFeatureExtractor) -> np.ndarray:
        """
        Extrai características LBP de um conjunto de imagens.
        
        Args:
            X_paths: Array com caminhos das imagens
            extractor: Extrator LBP configurado
            
        Returns:
            Matriz de características
        """
        features = []
        valid_indices = []
        
        for i, img_path in enumerate(X_paths):
            feat = extractor.extract_from_path(img_path)
            if feat is not None:
                features.append(feat)
                valid_indices.append(i)
        
        if not features:
            raise ValueError("Nenhuma característica válida extraída")
        
        return np.array(features), np.array(valid_indices)
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           n_estimators: int = 100, max_depth: int = 10,
                           min_samples_split: int = 2, min_samples_leaf: int = 1) -> RandomForestClassifier:
        """
        Treina um modelo Random Forest.
        
        Args:
            X_train: Características de treino
            y_train: Rótulos de treino
            n_estimators: Número de árvores
            max_depth: Profundidade máxima
            min_samples_split: Mínimo de amostras para dividir um nó
            min_samples_leaf: Mínimo de amostras em uma folha
            
        Returns:
            Modelo treinado
        """
        logger.info(f"Treinando Random Forest com:")
        logger.info(f"  n_estimators={n_estimators}")
        logger.info(f"  max_depth={max_depth}")
        logger.info(f"  min_samples_split={min_samples_split}")
        logger.info(f"  min_samples_leaf={min_samples_leaf}")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_SEED,
            n_jobs=-1,  # Usar todos os cores
            verbose=0
        )
        
        model.fit(X_train, y_train)
        logger.info("Treinamento concluído!")
        
        return model
    
    def evaluate_model(self, model: RandomForestClassifier, 
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Avalia o modelo com múltiplas métricas.
        
        Args:
            model: Modelo treinado
            X_test: Características de teste
            y_test: Rótulos de teste
            
        Returns:
            Dicionário com as métricas
        """
        # Predições
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }
        
        # Calcular confiança média para cada classe
        metrics['mean_confidence_class_0'] = np.mean(y_proba[y_test == 0][:, 0])
        metrics['mean_confidence_class_1'] = np.mean(y_proba[y_test == 1][:, 1])
        
        logger.info(f"Métricas de avaliação:")
        logger.info(f"  Acurácia: {metrics['accuracy']:.4f}")
        logger.info(f"  Precisão: {metrics['precision']:.4f}")
        logger.info(f"  Revocação: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Path):
        """Plota e salva a matriz de confusão."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Sem Humano', 'Com Humano'],
                   yticklabels=['Sem Humano', 'Com Humano'])
        plt.title('Matriz de Confusão')
        plt.ylabel('Real')
        plt.xlabel('Predito')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def grid_search(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   param_grid: Dict = None,
                   cv_folds: int = 5) -> Tuple[RandomForestClassifier, Dict]:
        """
        Realiza busca em grade de hiperparâmetros usando GridSearchCV (paralelizado).
        
        Args:
            X_train: Características de treino
            y_train: Rótulos de treino
            X_val: Características de validação
            y_val: Rótulos de validação
            param_grid: Grade de parâmetros para testar
            cv_folds: Número de folds para validação cruzada
            
        Returns:
            Melhor modelo e dicionário com resultados
        """
        if param_grid is None:
            param_grid = RF_CONFIG
        
        total_combinations = (len(param_grid['n_estimators']) * 
                            len(param_grid['max_depth']) *
                            len(param_grid['min_samples_split']) *
                            len(param_grid['min_samples_leaf']))
        
        logger.info(f"Iniciando GridSearchCV com {total_combinations} combinações")
        logger.info(f"Usando {cv_folds}-fold CV com paralelização (n_jobs=-1)")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
        
        base_model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2,
            refit=True
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        results = []
        for i in range(len(grid_search.cv_results_['params'])):
            result = {
                **grid_search.cv_results_['params'][i],
                'val_accuracy': grid_search.cv_results_['mean_test_score'][i],
                'val_f1': grid_search.cv_results_['mean_test_score'][i]
            }
            results.append(result)
        
        results.sort(key=lambda x: x['val_accuracy'], reverse=True)
        
        val_metrics = self.evaluate_model(best_model, X_val, y_val)
        
        logger.info(f"\nMelhor modelo encontrado:")
        logger.info(f"  CV Score: {grid_search.best_score_:.4f}")
        logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"  Parâmetros: {best_params}")
        
        return best_model, results
    
    def save_model_and_results(self, model: RandomForestClassifier, 
                              results: List[Dict], 
                              test_metrics: Dict,
                              experiment_name: str = None):
        """
        Salva o modelo e os resultados do treinamento.
        
        Args:
            model: Modelo treinado
            results: Resultados da busca em grade
            test_metrics: Métricas no conjunto de teste
            experiment_name: Nome do experimento
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salvar modelo
        model_path = self.models_dir / f"model_{experiment_name}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Modelo salvo em: {model_path}")
        
        # Salvar resultados
        results_path = self.reports_dir / f"results_{experiment_name}.json"
        with open(results_path, 'w') as f:
            json.dump({
                'grid_search_results': results,
                'test_metrics': test_metrics,
                'best_params': results[0] if results else None
            }, f, indent=4)
        logger.info(f"Resultados salvos em: {results_path}")
        
        return model_path, results_path

def main():
    """Função principal para treinamento."""
    
    # Carregar dados
    logger.info("Carregando dados...")
    loader = HumanDatasetLoader()
    data = loader.load_data()
    
    X_train_paths, y_train = data['train']
    X_val_paths, y_val = data['val']
    X_test_paths, y_test = data['test']
    
    # Inicializar extrator LBP
    extractor = LBPFeatureExtractor(
        radius=1,
        n_points=8,
        method='uniform'
    )
    
    # Extrair características
    logger.info("Extraindo características LBP...")
    X_train, train_idx = extractor.extract_batch(X_train_paths)
    y_train = y_train[train_idx]
    
    X_val, val_idx = extractor.extract_batch(X_val_paths)
    y_val = y_val[val_idx]
    
    X_test, test_idx = extractor.extract_batch(X_test_paths)
    y_test = y_test[test_idx]
    
    logger.info(f"Dimensões das características: {X_train.shape}")
    
    # Treinador
    trainer = ModelTrainer()
    
    # Busca em grade
    best_model, grid_results = trainer.grid_search(
        X_train, y_train,
        X_val, y_val
    )
    
    # Avaliar no teste
    test_metrics = trainer.evaluate_model(best_model, X_test, y_test)
    
    # Plotar matriz de confusão
    cm = np.array(test_metrics['confusion_matrix'])
    trainer.plot_confusion_matrix(
        cm, 
        trainer.reports_dir / f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    
    # Salvar modelo e resultados
    trainer.save_model_and_results(best_model, grid_results, test_metrics)
    
    logger.info("Treinamento concluído com sucesso!")

if __name__ == "__main__":
    main()