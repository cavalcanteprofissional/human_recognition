"""
Registro de modelos para comparação e seleção.
Gerencia todos os modelos, métricas e resultados de treinamento.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass, field, asdict
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import ADVANCED_MODELS_CONFIG, RANDOM_SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Métricas de um modelo."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    confusion_matrix: List[List[int]] = field(default_factory=list)
    training_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ModelMetrics":
        return cls(**data)


@dataclass
class ModelResult:
    """Resultado completo de um modelo treinado."""
    model_name: str
    model_type: str
    best_params: Dict[str, Any] = field(default_factory=dict)
    cv_metrics: ModelMetrics = field(default_factory=ModelMetrics)
    cv_std: Dict[str, float] = field(default_factory=dict)
    val_metrics: ModelMetrics = field(default_factory=ModelMetrics)
    test_metrics: ModelMetrics = field(default_factory=ModelMetrics)
    model: Optional[BaseEstimator] = None
    
    def to_dict(self, include_model: bool = False) -> Dict:
        result = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "best_params": self.best_params,
            "cv_metrics": self.cv_metrics.to_dict(),
            "cv_std": self.cv_std,
            "val_metrics": self.val_metrics.to_dict(),
            "test_metrics": self.test_metrics.to_dict()
        }
        return result


class ModelRegistry:
    """
    Registro centralizado de modelos.
    Gerencia criação, treinamento e comparação de múltiplos modelos.
    """
    
    _model_classes: Dict[str, Type[BaseEstimator]] = {}
    _model_configs: Dict[str, Dict] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseEstimator], config: Dict = None):
        """Registra um modelo no registry."""
        cls._model_classes[name] = model_class
        if config:
            cls._model_configs[name] = config
        logger.info(f"Modelo registrado: {name}")
    
    @classmethod
    def get_model_class(cls, name: str) -> Optional[Type[BaseEstimator]]:
        """Obtém a classe de um modelo pelo nome."""
        return cls._model_classes.get(name)
    
    @classmethod
    def get_config(cls, name: str) -> Dict:
        """Obtém a configuração de hiperparâmetros de um modelo."""
        return cls._model_configs.get(name, {})
    
    @classmethod
    def list_models(cls) -> List[str]:
        """Lista todos os modelos registrados."""
        return list(cls._model_classes.keys())
    
    @classmethod
    def create_model(cls, name: str, params: Dict = None) -> Optional[BaseEstimator]:
        """
        Cria uma instância do modelo com os parâmetros especificados.
        
        Args:
            name: Nome do modelo
            params: Parâmetros do modelo
            
        Returns:
            Instância do modelo
        """
        model_class = cls.get_model_class(name)
        if model_class is None:
            logger.error(f"Modelo não encontrado: {name}")
            return None
        
        params = params or {}
        
        try:
            if name == "svm":
                params["probability"] = True
                return model_class(random_state=RANDOM_SEED, **params)
            elif name in ["random_forest", "gradient_boosting", "mlp"]:
                return model_class(random_state=RANDOM_SEED, **params)
            elif name in ["xgboost", "lightgbm"]:
                params["random_state"] = RANDOM_SEED
                if name == "lightgbm" and "verbose" not in params:
                    params["verbose"] = -1
                return model_class(**params)
            elif name in ["knn", "logistic_regression"]:
                return model_class(**params)
            else:
                return model_class(**params)
        except Exception as e:
            logger.error(f"Erro ao criar modelo {name}: {e}")
            return None


def _register_default_models():
    """Registra os modelos padrão do sklearn."""
    ModelRegistry.register("random_forest", RandomForestClassifier, 
                          ADVANCED_MODELS_CONFIG.get("random_forest", {}))
    ModelRegistry.register("gradient_boosting", GradientBoostingClassifier,
                          ADVANCED_MODELS_CONFIG.get("gradient_boosting", {}))
    ModelRegistry.register("svm", SVC,
                          ADVANCED_MODELS_CONFIG.get("svm", {}))
    ModelRegistry.register("knn", KNeighborsClassifier,
                          ADVANCED_MODELS_CONFIG.get("knn", {}))
    ModelRegistry.register("logistic_regression", LogisticRegression,
                          ADVANCED_MODELS_CONFIG.get("logistic_regression", {}))
    ModelRegistry.register("mlp", MLPClassifier,
                          ADVANCED_MODELS_CONFIG.get("mlp", {}))


def register_xgboost():
    """Tenta registrar o XGBoost se disponível."""
    try:
        from xgboost import XGBClassifier
        ModelRegistry.register("xgboost", XGBClassifier,
                              ADVANCED_MODELS_CONFIG.get("xgboost", {}))
        logger.info("XGBoost registrado com sucesso")
        return True
    except ImportError:
        logger.warning("XGBoost não instalado. Use: pip install xgboost")
        return False


def register_lightgbm():
    """Tenta registrar o LightGBM se disponível."""
    try:
        from lightgbm import LGBMClassifier
        ModelRegistry.register("lightgbm", LGBMClassifier,
                              ADVANCED_MODELS_CONFIG.get("lightgbm", {}))
        logger.info("LightGBM registrado com sucesso")
        return True
    except ImportError:
        logger.warning("LightGBM não instalado. Use: pip install lightgbm")
        return False


class ModelComparison:
    """
    Comparação de múltiplos modelos treinados.
    """
    
    def __init__(self, results_dir: Path = None):
        self.results: List[ModelResult] = []
        self.results_dir = results_dir or Path("reports")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def add_result(self, result: ModelResult):
        """Adiciona resultado de um modelo."""
        self.results.append(result)
    
    def get_ranking(self, metric: str = "accuracy", 
                   dataset: str = "val") -> List[ModelResult]:
        """
        Retorna ranking dos modelos por métrica.
        
        Args:
            metric: Métrica para ordenação
            dataset: Dataset ('cv', 'val', 'test')
            
        Returns:
            Lista ordenada de resultados
        """
        def get_metric_value(result: ModelResult) -> float:
            if dataset == "cv":
                return getattr(result.cv_metrics, metric, 0)
            elif dataset == "val":
                return getattr(result.val_metrics, metric, 0)
            elif dataset == "test":
                return getattr(result.test_metrics, metric, 0)
            return 0
        
        return sorted(self.results, key=get_metric_value, reverse=True)
    
    def get_best_model(self, metric: str = "accuracy", 
                      dataset: str = "val") -> Optional[ModelResult]:
        """Retorna o melhor modelo por métrica."""
        ranking = self.get_ranking(metric, dataset)
        return ranking[0] if ranking else None
    
    def to_dict(self) -> Dict:
        """Converte para dicionário."""
        return {
            "results": [r.to_dict() for r in self.results],
            "ranking": {
                "val_accuracy": [r.model_name for r in self.get_ranking("accuracy", "val")],
                "test_accuracy": [r.model_name for r in self.get_ranking("accuracy", "test")],
                "test_f1": [r.model_name for r in self.get_ranking("f1_score", "test")]
            }
        }
    
    def save(self, filename: str = None) -> Path:
        """Salva resultados em JSON."""
        if filename is None:
            filename = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.results_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comparação salva em: {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> "ModelComparison":
        """Carrega resultados de JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        comparison = cls()
        for result_data in data["results"]:
            result = ModelResult(
                model_name=result_data["model_name"],
                model_type=result_data["model_type"],
                best_params=result_data["best_params"],
                cv_metrics=ModelMetrics.from_dict(result_data["cv_metrics"]),
                cv_std=result_data["cv_std"],
                val_metrics=ModelMetrics.from_dict(result_data["val_metrics"]),
                test_metrics=ModelMetrics.from_dict(result_data["test_metrics"])
            )
            comparison.add_result(result)
        
        return comparison
    
    def summary(self) -> str:
        """Gera resumo textual dos resultados."""
        lines = ["=" * 80, "COMPARAÇÃO DE MODELOS", "=" * 80, ""]
        
        header = f"{'Modelo':<25} {'CV Acc':>10} {'Val Acc':>10} {'Test Acc':>10} {'Test F1':>10} {'Tempo (s)':>10}"
        lines.append(header)
        lines.append("-" * 80)
        
        for result in self.get_ranking("accuracy", "val"):
            cv_acc = f"{result.cv_metrics.accuracy:.4f}"
            val_acc = f"{result.val_metrics.accuracy:.4f}"
            test_acc = f"{result.test_metrics.accuracy:.4f}"
            test_f1 = f"{result.test_metrics.f1_score:.4f}"
            time_s = f"{result.test_metrics.training_time:.2f}"
            
            line = f"{result.model_name:<25} {cv_acc:>10} {val_acc:>10} {test_acc:>10} {test_f1:>10} {time_s:>10}"
            lines.append(line)
        
        lines.append("=" * 80)
        
        best = self.get_best_model()
        if best:
            lines.append(f"Melhor modelo: {best.model_name}")
            lines.append(f"  Test Accuracy: {best.test_metrics.accuracy:.4f}")
            lines.append(f"  Test F1-Score: {best.test_metrics.f1_score:.4f}")
        
        return "\n".join(lines)


_register_default_models()
register_xgboost()
register_lightgbm()
