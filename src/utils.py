import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Dict
import cv2

from src.config import REPORTS_DIR

def plot_training_results(results_path: Path):
    """
    Plota os resultados do treinamento.
    
    Args:
        results_path: Caminho para o arquivo de resultados JSON
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    grid_results = results['grid_search_results']
    
    # Extrair parâmetros e métricas
    n_estimators = [r['n_estimators'] for r in grid_results]
    max_depths = [r['max_depth'] for r in grid_results]
    accuracies = [r['val_accuracy'] for r in grid_results]
    f1_scores = [r['val_f1'] for r in grid_results]
    
    # Criar figura
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gráfico de acurácia por n_estimators
    ax1 = axes[0, 0]
    for depth in set(max_depths):
        indices = [i for i, d in enumerate(max_depths) if d == depth]
        ax1.scatter([n_estimators[i] for i in indices], 
                   [accuracies[i] for i in indices], 
                   label=f'max_depth={depth}', alpha=0.6)
    ax1.set_xlabel('n_estimators')
    ax1.set_ylabel('Acurácia')
    ax1.set_title('Acurácia por n_estimators')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de F1 por max_depth
    ax2 = axes[0, 1]
    depths = sorted(set(max_depths))
    mean_f1 = [np.mean([f1_scores[i] for i, d in enumerate(max_depths) if d == depth]) 
               for depth in depths]
    ax2.plot(depths, mean_f1, 'bo-')
    ax2.set_xlabel('max_depth')
    ax2.set_ylabel('F1-Score médio')
    ax2.set_title('F1-Score médio por max_depth')
    ax2.grid(True, alpha=0.3)
    
    # Heatmap de parâmetros
    ax3 = axes[1, 0]
    param_matrix = np.zeros((len(set(n_estimators)), len(set(max_depths))))
    unique_n = sorted(set(n_estimators))
    unique_d = sorted(set(max_depths))
    
    for i, n in enumerate(unique_n):
        for j, d in enumerate(unique_d):
            accs = [accuracies[k] for k, (n_est, depth) in enumerate(zip(n_estimators, max_depths)) 
                   if n_est == n and depth == d]
            if accs:
                param_matrix[i, j] = np.mean(accs)
    
    im = ax3.imshow(param_matrix, cmap='viridis', aspect='auto')
    ax3.set_xticks(range(len(unique_d)))
    ax3.set_yticks(range(len(unique_n)))
    ax3.set_xticklabels(unique_d)
    ax3.set_yticklabels(unique_n)
    ax3.set_xlabel('max_depth')
    ax3.set_ylabel('n_estimators')
    ax3.set_title('Acurácia média (parâmetros)')
    plt.colorbar(im, ax=ax3)
    
    # Melhores resultados
    ax4 = axes[1, 1]
    top_10 = grid_results[:10]
    params = [f"n={r['n_estimators']}, d={r['max_depth']}" for r in top_10]
    accs_top = [r['val_accuracy'] for r in top_10]
    
    y_pos = np.arange(len(params))
    ax4.barh(y_pos, accs_top)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(params)
    ax4.set_xlabel('Acurácia')
    ax4.set_title('Top 10 Configurações')
    ax4.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'training_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_sample_comparison():
    """
    Cria uma imagem comparando diferentes filtros.
    """
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Não foi possível capturar imagem")
        return
    
    filters = ['cartoon', 'edges', 'colormap', 'stylized', 'pencil']
    from src.real_time_detector import HumanDetector
    detector = HumanDetector()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Cada filtro
    for i, filter_name in enumerate(filters):
        filtered = detector.apply_creative_filter(frame, filter_name)
        axes[i+1].imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
        axes[i+1].set_title(filter_name.capitalize())
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'filter_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()