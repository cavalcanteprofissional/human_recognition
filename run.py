#!/usr/bin/env python3
"""
Script principal para executar o projeto de reconhecimento de silhueta humana.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from src.data_loader import HumanDatasetLoader
from src.train import main as train_main
from src.real_time_detector import main as detect_main
from src.utils import plot_training_results, create_sample_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def setup_project():
    """Configura o projeto baixando e preparando os dados."""
    logger.info("Configurando projeto...")
    loader = HumanDatasetLoader()
    data = loader.load_data()
    logger.info("Projeto configurado com sucesso!")

def run_dashboard():
    """Executa o dashboard Streamlit."""
    dashboard_path = Path(__file__).parent / "src" / "dashboard.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path)]
    subprocess.run(cmd)

def run_advanced_training(models: str = None, cv_folds: int = 5, no_ensemble: bool = False):
    """
    Executa treinamento avançado com múltiplos modelos.
    
    Args:
        models: Lista de modelos separados por vírgula (ou None para todos)
        cv_folds: Número de folds na validação cruzada
        no_ensemble: Se True, não cria ensemble
    """
    from src.train_advanced import AdvancedTrainer
    from src.model_registry import ModelRegistry
    from src.feature_extractor import LBPFeatureExtractor
    
    logger.info("=" * 60)
    logger.info("TREINAMENTO AVANÇADO")
    logger.info("=" * 60)
    
    logger.info("Carregando dados...")
    loader = HumanDatasetLoader()
    data = loader.load_data()
    
    X_train_paths, y_train = data["train"]
    X_val_paths, y_val = data["val"]
    X_test_paths, y_test = data["test"]
    
    logger.info("Extraindo características LBP...")
    extractor = LBPFeatureExtractor(radius=1, n_points=8, method="uniform")
    
    trainer = AdvancedTrainer(cv_folds=cv_folds)
    
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_data(
        X_train_paths, y_train,
        X_val_paths, y_val,
        X_test_paths, y_test,
        extractor
    )
    
    if models:
        model_names = [m.strip() for m in models.split(",")]
        logger.info(f"Modelos selecionados: {model_names}")
    else:
        model_names = ModelRegistry.list_models()
        logger.info(f"Treinando todos os {len(model_names)} modelos")
    
    comparison = trainer.train_all_models(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        model_names=model_names,
        include_ensemble=not no_ensemble
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

def compare_models(results_path: str = None):
    """
    Compara modelos a partir de resultados salvos.
    
    Args:
        results_path: Caminho para o arquivo JSON de comparação
    """
    from src.model_registry import ModelComparison
    
    if results_path:
        comparison = ModelComparison.load(Path(results_path))
    else:
        reports_dir = Path("reports")
        json_files = list(reports_dir.glob("model_comparison_*.json"))
        if not json_files:
            logger.error("Nenhum arquivo de comparação encontrado")
            return
        latest = max(json_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Carregando: {latest}")
        comparison = ModelComparison.load(latest)
    
    print(comparison.summary())

def main():
    parser = argparse.ArgumentParser(description='Human Recognition Project')
    
    # Setup e dados
    parser.add_argument('--setup', action='store_true', 
                       help='Baixar e preparar dados')
    
    # Treinamento básico
    parser.add_argument('--train', action='store_true', 
                       help='Treinar modelo (Random Forest simples)')
    
    # Treinamento avançado
    parser.add_argument('--train-advanced', action='store_true',
                       help='Treinar múltiplos modelos com validação cruzada')
    parser.add_argument('--models', type=str, default=None,
                       help='Modelos para treinar (separados por vírgula). Ex: random_forest,xgboost,svm')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Número de folds na validação cruzada (padrão: 5)')
    parser.add_argument('--no-ensemble', action='store_true',
                       help='Não criar ensemble de modelos')
    
    # Comparação
    parser.add_argument('--compare-models', action='store_true',
                       help='Comparar modelos treinados')
    parser.add_argument('--results', type=str, default=None,
                       help='Arquivo JSON com resultados para comparar')
    
    # Detecção em tempo real
    parser.add_argument('--detect', action='store_true', 
                       help='Executar detecção em tempo real (terminal)')
    parser.add_argument('--filter', type=str, default='cartoon',
                       choices=['cartoon', 'edges', 'colormap', 'stylized', 'pencil', 'none'],
                       help='Filtro a ser aplicado (padrão: cartoon)')
    parser.add_argument('--source', type=str, default='webcam',
                       choices=['webcam', 'yoosee'],
                       help='Fonte de vídeo (padrão: webcam)')
    parser.add_argument('--yoosee-ip', type=str, default=None,
                       help='IP da câmera Yoosee')
    parser.add_argument('--yoosee-stream', type=str, default='onvif1',
                       choices=['onvif1', 'onvif2', 'live', 'stream11', 'h264'],
                       help='Stream da Yoosee (padrão: onvif1)')
    
    # Auto-discovery
    parser.add_argument('--auto-find-yoosee', action='store_true',
                       help='Encontrar IP da câmera Yoosee automaticamente na rede')
    
    # Dashboard
    parser.add_argument('--dashboard', action='store_true',
                       help='Executar dashboard Streamlit')
    
    # Análise
    parser.add_argument('--analyze', type=str,
                       help='Analisar resultados de treinamento (caminho do JSON)')
    parser.add_argument('--compare-filters', action='store_true',
                       help='Criar comparação de filtros')
    
    # Listar modelos disponíveis
    parser.add_argument('--list-models', action='store_true',
                       help='Listar modelos disponíveis para treinamento')
    
    args = parser.parse_args()
    
    # Listar modelos
    if args.list_models:
        from src.model_registry import ModelRegistry
        print("\nModelos disponíveis:")
        for model in ModelRegistry.list_models():
            config = ModelRegistry.get_config(model)
            n_params = len(config) if config else 0
            print(f"  - {model} ({n_params} hiperparâmetros)")
        return
    
    # Auto-discovery
    if args.auto_find_yoosee:
        from src.config import find_and_update_yoosee_ip
        logger.info("Buscando câmera Yoosee na rede...")
        ip, port, stream = find_and_update_yoosee_ip()
        if ip:
            logger.info(f"Câmera encontrada: {ip}:{port}, stream: {stream}")
        else:
            logger.warning("Câmera não encontrada na rede.")
        return
    
    # Setup
    if args.setup:
        setup_project()
    
    # Treinamento básico
    if args.train:
        train_main()
    
    # Treinamento avançado
    if args.train_advanced:
        run_advanced_training(
            models=args.models,
            cv_folds=args.cv_folds,
            no_ensemble=args.no_ensemble
        )
    
    # Comparar modelos
    if args.compare_models:
        compare_models(args.results)
    
    # Detecção em tempo real
    if args.detect:
        from src.real_time_detector import HumanDetector
        detector = HumanDetector()
        detector.run(
            filter_type=args.filter,
            source=args.source,
            yoosee_ip=args.yoosee_ip,
            yoosee_stream=args.yoosee_stream,
            auto_find_yoosee=args.auto_find_yoosee
        )
    
    # Dashboard
    if args.dashboard:
        run_dashboard()
    
    # Análise
    if args.analyze:
        plot_training_results(Path(args.analyze))
    
    if args.compare_filters:
        create_sample_comparison()
    
    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()