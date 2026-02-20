"""
Pipeline avançado de treinamento com múltiplos modelos e validação cruzada.

Implementa:
- Divisão 70/15/15 (treino/validação/teste)
- Validação cruzada com 5 folds internos
- Grid Search para hiperparâmetros
- Comparação de 9 modelos diferentes
- Ensemble Voting
- Visualização de resultados
"""

import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
import joblib
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    MODELS_DIR, REPORTS_DIR, RANDOM_SEED,
    TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE,
    CV_FOLDS, CV_SCORING, ADVANCED_MODELS_CONFIG, ENSEMBLE_BASE_MODELS
)
from src.feature_extractor import LBPFeatureExtractor
from src.data_loader import HumanDatasetLoader
from src.model_registry import (
    ModelRegistry, ModelMetrics, ModelResult, ModelComparison
)
from src.ensemble import EnsembleBuilder, create_default_ensemble, evaluate_ensemble

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AdvancedTrainer:
    """
    Treinador avançado com múltiplos modelos e validação cruzada.
    """
    
    def __init__(self, 
                 cv_folds: int = CV_FOLDS,
                 random_state: int = RANDOM_SEED,
                 models_dir: Path = None,
                 reports_dir: Path = None):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.models_dir = models_dir or MODELS_DIR
        self.reports_dir = reports_dir or REPORTS_DIR
        self.comparison = ModelComparison(self.reports_dir)
        self.trained_models: Dict[str, Any] = {}
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, 
                    X_train_paths: np.ndarray, y_train: np.ndarray,
                    X_val_paths: np.ndarray, y_val: np.ndarray,
                    X_test_paths: np.ndarray, y_test: np.ndarray,
                    extractor: LBPFeatureExtractor) -> Tuple[np.ndarray, ...]:
        """
        Extrai características LBP dos dados.
        """
        logger.info("Extraindo características LBP...")
        
        X_train, train_idx = extractor.extract_batch(X_train_paths)
        y_train = y_train[train_idx]
        
        X_val, val_idx = extractor.extract_batch(X_val_paths)
        y_val = y_val[val_idx]
        
        X_test, test_idx = extractor.extract_batch(X_test_paths)
        y_test = y_test[test_idx]
        
        logger.info(f"Dimensões: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_proba: np.ndarray = None) -> ModelMetrics:
        """Calcula métricas de avaliação."""
        metrics = ModelMetrics()
        metrics.accuracy = accuracy_score(y_true, y_pred)
        metrics.precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
        metrics.recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
        metrics.f1_score = f1_score(y_true, y_pred, average="binary", zero_division=0)
        
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics.auc_roc = roc_auc_score(y_true, y_proba[:, 1])
            except:
                metrics.auc_roc = 0.0
        
        cm = confusion_matrix(y_true, y_pred)
        metrics.confusion_matrix = cm.tolist()
        
        return metrics
    
    def train_single_model(self,
                          model_name: str,
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          param_grid: Dict = None) -> Optional[ModelResult]:
        """
        Treina um único modelo com Grid Search CV.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"TREINANDO: {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        model_class = ModelRegistry.get_model_class(model_name)
        if model_class is None:
            logger.warning(f"Modelo {model_name} não disponível")
            return None
        
        if param_grid is None:
            param_grid = ModelRegistry.get_config(model_name)
        
        if not param_grid:
            logger.warning(f"Sem param_grid para {model_name}, usando defaults")
            param_grid = {}
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                            random_state=self.random_state)
        
        base_model = ModelRegistry.create_model(model_name)
        if base_model is None:
            return None
        
        start_time = time.time()
        
        if param_grid:
            logger.info(f"Grid Search com {self._count_combinations(param_grid)} combinações...")
            
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring=CV_SCORING,
                n_jobs=-1,
                verbose=1,
                refit=True
            )
            
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_results = grid_search.cv_results_
            
            cv_mean = cv_results["mean_test_score"][grid_search.best_index_]
            cv_std = cv_results["std_test_score"][grid_search.best_index_]
        else:
            logger.info("Treinando sem Grid Search...")
            best_model = base_model
            best_model.fit(X_train, y_train)
            best_params = {}
            
            cv_scores = cross_val_score(best_model, X_train, y_train, 
                                       cv=cv, scoring=CV_SCORING, n_jobs=-1)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        
        training_time = time.time() - start_time
        
        logger.info(f"Melhores parâmetros: {best_params}")
        logger.info(f"CV Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        y_val_pred = best_model.predict(X_val)
        y_val_proba = best_model.predict_proba(X_val) if hasattr(best_model, "predict_proba") else None
        val_metrics = self._compute_metrics(y_val, y_val_pred, y_val_proba)
        
        start_time = time.time()
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None
        test_metrics = self._compute_metrics(y_test, y_test_pred, y_test_proba)
        test_metrics.training_time = training_time
        
        logger.info(f"Validação - Acc: {val_metrics.accuracy:.4f}, F1: {val_metrics.f1_score:.4f}")
        logger.info(f"Teste     - Acc: {test_metrics.accuracy:.4f}, F1: {test_metrics.f1_score:.4f}")
        
        cv_metrics = ModelMetrics(accuracy=cv_mean)
        cv_std_dict = {"accuracy": cv_std}
        
        model_type = "ensemble" if model_name in ["voting_ensemble", "stacking_ensemble"] else "single"
        
        result = ModelResult(
            model_name=model_name,
            model_type=model_type,
            best_params=best_params,
            cv_metrics=cv_metrics,
            cv_std=cv_std_dict,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            model=best_model
        )
        
        self.trained_models[model_name] = best_model
        
        return result
    
    def _count_combinations(self, param_grid: Dict) -> int:
        """Conta número de combinações no grid."""
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count
    
    def train_all_models(self,
                        X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        model_names: List[str] = None,
                        include_ensemble: bool = True) -> ModelComparison:
        """
        Treina todos os modelos e retorna comparação.
        """
        if model_names is None:
            model_names = ModelRegistry.list_models()
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"INICIANDO TREINAMENTO DE {len(model_names)} MODELOS")
        logger.info(f"{'#'*60}\n")
        
        for model_name in model_names:
            result = self.train_single_model(
                model_name=model_name,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test
            )
            
            if result:
                self.comparison.add_result(result)
        
        if include_ensemble and len(self.trained_models) >= 2:
            logger.info(f"\n{'='*60}")
            logger.info("CRIANDO ENSEMBLE (VOTING)")
            logger.info(f"{'='*60}")
            
            ensemble_models = [m for m in ENSEMBLE_BASE_MODELS if m in self.trained_models]
            
            if len(ensemble_models) >= 2:
                ensemble = create_default_ensemble(
                    self.trained_models,
                    ensemble_type="voting",
                    model_names=ensemble_models[:5]
                )
                
                try:
                    ensemble.fit(X_train, y_train)
                    
                    y_val_pred = ensemble.predict(X_val)
                    y_val_proba = ensemble.predict_proba(X_val)
                    val_metrics = self._compute_metrics(y_val, y_val_pred, y_val_proba)
                    
                    y_test_pred = ensemble.predict(X_test)
                    y_test_proba = ensemble.predict_proba(X_test)
                    test_metrics = self._compute_metrics(y_test, y_test_pred, y_test_proba)
                    
                    logger.info(f"Ensemble - Val Acc: {val_metrics.accuracy:.4f}, Test Acc: {test_metrics.accuracy:.4f}")
                    
                    ensemble_result = ModelResult(
                        model_name="voting_ensemble",
                        model_type="ensemble",
                        best_params={"base_models": ensemble_models[:5]},
                        cv_metrics=ModelMetrics(),
                        cv_std={},
                        val_metrics=val_metrics,
                        test_metrics=test_metrics,
                        model=ensemble.ensemble
                    )
                    
                    self.comparison.add_result(ensemble_result)
                    self.trained_models["voting_ensemble"] = ensemble.ensemble
                    
                except Exception as e:
                    logger.error(f"Erro ao criar ensemble: {e}")
        
        best = self.comparison.get_best_model(metric="accuracy", dataset="val")
        if best:
            self.best_model = best.model
            self.best_model_name = best.model_name
            logger.info(f"\n{'*'*60}")
            logger.info(f"MELHOR MODELO: {best.model_name}")
            logger.info(f"Test Accuracy: {best.test_metrics.accuracy:.4f}")
            logger.info(f"Test F1-Score: {best.test_metrics.f1_score:.4f}")
            logger.info(f"{'*'*60}\n")
        
        return self.comparison
    
    def plot_comparison(self, save_path: Path = None):
        """Gera gráficos de comparação dos modelos."""
        if not self.comparison.results:
            logger.warning("Sem resultados para plotar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        model_names = [r.model_name for r in self.comparison.results]
        val_accs = [r.val_metrics.accuracy for r in self.comparison.results]
        test_accs = [r.test_metrics.accuracy for r in self.comparison.results]
        test_f1s = [r.test_metrics.f1_score for r in self.comparison.results]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax1 = axes[0, 0]
        bars1 = ax1.bar(x - width/2, val_accs, width, label='Validação', color='skyblue')
        bars2 = ax1.bar(x + width/2, test_accs, width, label='Teste', color='salmon')
        ax1.set_ylabel('Acurácia')
        ax1.set_title('Acurácia por Modelo')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        ax1.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='Target 85%')
        
        ax2 = axes[0, 1]
        ax2.bar(model_names, test_f1s, color='lightgreen')
        ax2.set_ylabel('F1-Score')
        ax2.set_title('F1-Score no Teste')
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.set_ylim(0, 1.1)
        
        ax3 = axes[1, 0]
        cv_stds = [r.cv_std.get("accuracy", 0) for r in self.comparison.results]
        ax3.bar(model_names, cv_stds, color='lightcoral')
        ax3.set_ylabel('Std (CV)')
        ax3.set_title('Desvio Padrão na Validação Cruzada')
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        
        ax4 = axes[1, 1]
        times = [r.test_metrics.training_time for r in self.comparison.results]
        ax4.barh(model_names, times, color='plum')
        ax4.set_xlabel('Tempo (s)')
        ax4.set_title('Tempo de Treinamento')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.reports_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico salvo em: {save_path}")
    
    def plot_confusion_matrices(self, save_path: Path = None):
        """Plota matrizes de confusão dos melhores modelos."""
        top_models = self.comparison.get_ranking("accuracy", "test")[:4]
        
        if not top_models:
            return
        
        n_models = len(top_models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, result in zip(axes, top_models):
            cm = np.array(result.test_metrics.confusion_matrix)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Sem Humano', 'Com Humano'],
                       yticklabels=['Sem Humano', 'Com Humano'])
            ax.set_title(f"{result.model_name}\nAcc: {result.test_metrics.accuracy:.2%}")
            ax.set_ylabel('Real')
            ax.set_xlabel('Predito')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.reports_dir / f"confusion_matrices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Matrizes de confusão salvas em: {save_path}")
    
    def save_best_model(self, filename: str = None) -> Path:
        """Salva o melhor modelo."""
        if self.best_model is None:
            logger.error("Nenhum modelo treinado")
            return None
        
        if filename is None:
            filename = f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        filepath = self.models_dir / filename
        joblib.dump({
            "model": self.best_model,
            "model_name": self.best_model_name,
            "timestamp": datetime.now().isoformat()
        }, filepath)
        
        logger.info(f"Melhor modelo salvo em: {filepath}")
        return filepath
    
    def save_results(self) -> Tuple[Path, Path, Path]:
        """Salva todos os resultados."""
        comparison_path = self.comparison.save()
        
        model_path = self.save_best_model()
        
        self.plot_comparison()
        self.plot_confusion_matrices()
        
        return comparison_path, model_path, self.reports_dir


def main():
    """Função principal para treinamento avançado."""
    logger.info("=" * 60)
    logger.info("TREINAMENTO AVANÇADO - MÚLTIPLOS MODELOS")
    logger.info("=" * 60)
    
    logger.info("Carregando dados...")
    loader = HumanDatasetLoader()
    data = loader.load_data()
    
    X_train_paths, y_train = data["train"]
    X_val_paths, y_val = data["val"]
    X_test_paths, y_test = data["test"]
    
    logger.info("Divisão dos dados:")
    logger.info(f"  Treino: {len(X_train_paths)} ({len(X_train_paths)/len(X_train_paths)*100:.0f}%)")
    logger.info(f"  Validação: {len(X_val_paths)} ({len(X_val_paths)/len(X_train_paths)*100:.0f}%)")
    logger.info(f"  Teste: {len(X_test_paths)} ({len(X_test_paths)/len(X_train_paths)*100:.0f}%)")
    
    extractor = LBPFeatureExtractor(radius=1, n_points=8, method="uniform")
    
    trainer = AdvancedTrainer(cv_folds=CV_FOLDS)
    
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_data(
        X_train_paths, y_train,
        X_val_paths, y_val,
        X_test_paths, y_test,
        extractor
    )
    
    comparison = trainer.train_all_models(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        include_ensemble=True
    )
    
    print("\n" + comparison.summary())
    
    comparison_path, model_path, reports_dir = trainer.save_results()
    
    logger.info("\n" + "=" * 60)
    logger.info("TREINAMENTO CONCLUÍDO!")
    logger.info("=" * 60)
    logger.info(f"Comparação: {comparison_path}")
    logger.info(f"Melhor modelo: {model_path}")
    logger.info(f"Gráficos: {reports_dir}")
    
    return trainer


if __name__ == "__main__":
    main()
