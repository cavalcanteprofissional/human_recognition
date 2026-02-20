"""
Ensemble de modelos usando Voting e Stacking.
Combina múltiplos classificadores para melhor performance.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import RANDOM_SEED, ENSEMBLE_BASE_MODELS
from src.model_registry import ModelRegistry, ModelMetrics, ModelResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleBuilder:
    """
    Construtor de ensembles a partir de modelos treinados.
    """
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.trained_models: Dict[str, BaseEstimator] = {}
    
    def add_trained_model(self, name: str, model: BaseEstimator):
        """Adiciona um modelo treinado ao pool."""
        self.trained_models[name] = model
        logger.info(f"Modelo adicionado ao ensemble pool: {name}")
    
    def build_voting_ensemble(self, 
                             model_names: List[str] = None,
                             voting: str = "soft",
                             weights: List[float] = None) -> Optional[VotingClassifier]:
        """
        Constrói um VotingClassifier com os modelos especificados.
        
        Args:
            model_names: Lista de nomes dos modelos (usa todos se None)
            voting: 'hard' ou 'soft'
            weights: Pesos para cada modelo (opcional)
            
        Returns:
            VotingClassifier configurado
        """
        if model_names is None:
            model_names = list(self.trained_models.keys())
        
        estimators = []
        for name in model_names:
            if name in self.trained_models:
                model = self.trained_models[name]
                estimators.append((name, clone(model)))
            else:
                logger.warning(f"Modelo não encontrado: {name}")
        
        if not estimators:
            logger.error("Nenhum modelo válido para criar ensemble")
            return None
        
        logger.info(f"Criando Voting ensemble com {len(estimators)} modelos: {[e[0] for e in estimators]}")
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights
        )
        
        return ensemble
    
    def build_stacking_ensemble(self,
                               model_names: List[str] = None,
                               final_estimator: BaseEstimator = None,
                               cv: int = 5) -> Optional[StackingClassifier]:
        """
        Constrói um StackingClassifier com os modelos especificados.
        
        Args:
            model_names: Lista de nomes dos modelos (usa todos se None)
            final_estimator: Modelo final para meta-learning (LogReg se None)
            cv: Número de folds para CV interna
            
        Returns:
            StackingClassifier configurado
        """
        if model_names is None:
            model_names = list(self.trained_models.keys())
        
        estimators = []
        for name in model_names:
            if name in self.trained_models:
                model = self.trained_models[name]
                estimators.append((name, clone(model)))
            else:
                logger.warning(f"Modelo não encontrado: {name}")
        
        if not estimators:
            logger.error("Nenhum modelo válido para criar ensemble")
            return None
        
        if final_estimator is None:
            final_estimator = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            )
        
        logger.info(f"Criando Stacking ensemble com {len(estimators)} modelos base")
        
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method="predict_proba"
        )
        
        return ensemble


class EnsembleModel:
    """
    Wrapper para ensemble com interface unificada.
    """
    
    def __init__(self, ensemble_type: str = "voting", **kwargs):
        """
        Args:
            ensemble_type: 'voting' ou 'stacking'
            **kwargs: Argumentos para o ensemble
        """
        self.ensemble_type = ensemble_type
        self.builder = EnsembleBuilder()
        self.ensemble: Optional[BaseEstimator] = None
        self.kwargs = kwargs
        self.is_fitted = False
    
    def add_base_model(self, name: str, model: BaseEstimator):
        """Adiciona modelo base."""
        self.builder.add_trained_model(name, model)
    
    def build(self) -> BaseEstimator:
        """Constrói o ensemble."""
        if self.ensemble_type == "voting":
            self.ensemble = self.builder.build_voting_ensemble(**self.kwargs)
        elif self.ensemble_type == "stacking":
            self.ensemble = self.builder.build_stacking_ensemble(**self.kwargs)
        else:
            raise ValueError(f"Tipo de ensemble desconhecido: {self.ensemble_type}")
        
        return self.ensemble
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleModel":
        """Treina o ensemble."""
        if self.ensemble is None:
            self.build()
        
        if self.ensemble is not None:
            logger.info(f"Treinando ensemble {self.ensemble_type}...")
            self.ensemble.fit(X, y)
            self.is_fitted = True
            logger.info("Ensemble treinado!")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições."""
        if not self.is_fitted:
            raise ValueError("Ensemble não treinado")
        return self.ensemble.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades."""
        if not self.is_fitted:
            raise ValueError("Ensemble não treinado")
        return self.ensemble.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Retorna acurácia."""
        if not self.is_fitted:
            raise ValueError("Ensemble não treinado")
        return self.ensemble.score(X, y)


def create_default_ensemble(trained_models: Dict[str, BaseEstimator],
                           ensemble_type: str = "voting",
                           model_names: List[str] = None) -> EnsembleModel:
    """
    Cria ensemble com configurações padrão.
    
    Args:
        trained_models: Dicionário de modelos treinados
        ensemble_type: 'voting' ou 'stacking'
        model_names: Lista de modelos para incluir
        
    Returns:
        EnsembleModel configurado e pronto para treinar
    """
    ensemble = EnsembleModel(ensemble_type=ensemble_type)
    
    for name, model in trained_models.items():
        ensemble.add_base_model(name, model)
    
    if model_names:
        if ensemble_type == "voting":
            ensemble.kwargs["model_names"] = model_names
        else:
            ensemble.kwargs["model_names"] = model_names
    
    return ensemble


def evaluate_ensemble(ensemble: EnsembleModel,
                     X_val: np.ndarray, y_val: np.ndarray,
                     X_test: np.ndarray = None, y_test: np.ndarray = None) -> Tuple[ModelMetrics, Optional[ModelMetrics]]:
    """
    Avalia ensemble em validação e teste.
    
    Returns:
        (val_metrics, test_metrics)
    """
    def _compute_metrics(X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        y_pred = ensemble.predict(X)
        y_proba = ensemble.predict_proba(X)
        
        metrics = ModelMetrics()
        metrics.accuracy = accuracy_score(y, y_pred)
        metrics.precision = precision_score(y, y_pred, average="binary", zero_division=0)
        metrics.recall = recall_score(y, y_pred, average="binary", zero_division=0)
        metrics.f1_score = f1_score(y, y_pred, average="binary", zero_division=0)
        
        try:
            from sklearn.metrics import roc_auc_score
            metrics.auc_roc = roc_auc_score(y, y_proba[:, 1])
        except:
            metrics.auc_roc = 0.0
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y, y_pred)
        metrics.confusion_matrix = cm.tolist()
        
        return metrics
    
    val_metrics = _compute_metrics(X_val, y_val)
    
    test_metrics = None
    if X_test is not None and y_test is not None:
        test_metrics = _compute_metrics(X_test, y_test)
    
    return val_metrics, test_metrics
