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

logging.basicConfig(level=logging.INFO)
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

def main():
    parser = argparse.ArgumentParser(description='Human Recognition Project')
    parser.add_argument('--setup', action='store_true', 
                       help='Baixar e preparar dados')
    parser.add_argument('--train', action='store_true', 
                       help='Treinar modelo')
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
    parser.add_argument('--analyze', type=str,
                       help='Analisar resultados de treinamento (caminho do JSON)')
    parser.add_argument('--compare-filters', action='store_true',
                       help='Criar comparação de filtros')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_project()
    
    if args.train:
        train_main()
    
    if args.detect:
        from src.real_time_detector import HumanDetector
        detector = HumanDetector()
        detector.run(
            filter_type=args.filter,
            source=args.source,
            yoosee_ip=args.yoosee_ip,
            yoosee_stream=args.yoosee_stream
        )
    
    if args.dashboard:
        run_dashboard()
    
    if args.analyze:
        plot_training_results(Path(args.analyze))
    
    if args.compare_filters:
        create_sample_comparison()
    
    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()