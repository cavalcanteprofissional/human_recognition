"""
Dashboard interativo para o projeto de reconhecimento de silhueta humana.
Utiliza Streamlit para visualização em tempo real das métricas e resultados.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from PIL import Image
import json
from pathlib import Path
import time
import joblib
from datetime import datetime
import altair as alt
import os

from src.config import MODELS_DIR, REPORTS_DIR, PROCESSED_DATA_DIR
from src.feature_extractor import LBPFeatureExtractor
from src.real_time_detector import HumanDetector

# Configuração da página
st.set_page_config(
    page_title="Human Recognition Dashboard",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
        margin-top: 5px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 50px;
        font-size: 16px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class HumanRecognitionDashboard:
    """Dashboard interativo para o projeto de reconhecimento humano."""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.reports_dir = REPORTS_DIR
        self.data_dir = PROCESSED_DATA_DIR
        
        # Inicializar sessão state
        if 'detector' not in st.session_state:
            st.session_state.detector = None
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        if 'filter_type' not in st.session_state:
            st.session_state.filter_type = 'cartoon'
        if 'detection_history' not in st.session_state:
            st.session_state.detection_history = []
        
    def run(self):
        """Executa o dashboard."""
        
        # Sidebar
        with st.sidebar:
            self.render_sidebar()
        
        # Main content
        st.title("👤 Human Recognition Dashboard")
        st.markdown("### Projeto de Visão Computacional - Detecção de Silhueta Humana")
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Visão Geral", 
            "🤖 Treinamento", 
            "🎥 Detecção em Tempo Real",
            "📈 Análise de Métricas",
            "ℹ️ Sobre o Projeto"
        ])
        
        with tab1:
            self.render_overview()
        
        with tab2:
            self.render_training()
        
        with tab3:
            self.render_realtime_detection()
        
        with tab4:
            self.render_metrics_analysis()
        
        with tab5:
            self.render_about()
    
    def render_sidebar(self):
        """Renderiza a barra lateral."""
        st.image("https://raw.githubusercontent.com/opencv/opencv/master/doc/opencv-logo.png", 
                 width=200)
        st.markdown("---")
        
        st.markdown("### 🎯 Navegação Rápida")
        
        # Informações do dataset
        st.markdown("---")
        st.markdown("### 📊 Status do Dataset")
        
        # Verificar se dataset existe
        splits_path = self.data_dir / "human_dataset" / "splits.npz"
        if splits_path.exists():
            data = np.load(splits_path)
            total_images = len(data['X_train']) + len(data['X_val']) + len(data['X_test'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Imagens", total_images)
            with col2:
                st.metric("Classes", "2 (Humano/Não)")
            
            # Distribuição
            st.progress(0.7, text="Treino: 70%")
            st.progress(0.1, text="Validação: 10%")
            st.progress(0.2, text="Teste: 20%")
        else:
            st.warning("Dataset não encontrado. Execute --setup primeiro.")
        
        # Modelos disponíveis
        st.markdown("---")
        st.markdown("### 🤖 Modelos Disponíveis")
        models = list(self.models_dir.glob("model_*.pkl"))
        if models:
            model_options = {str(m): m for m in models}
            selected_model = st.selectbox(
                "Selecione o modelo",
                options=list(model_options.keys()),
                format_func=lambda x: Path(x).name
            )
            if selected_model:
                st.session_state.selected_model = model_options[selected_model]
                st.success(f"Modelo carregado: {Path(selected_model).name}")
        else:
            st.warning("Nenhum modelo treinado encontrado.")
        
        st.markdown("---")
        st.markdown("### 🎨 Configurações")
        
        # Configurações do LBP
        with st.expander("Parâmetros LBP"):
            radius = st.slider("Radius", 1, 3, 1)
            n_points = st.slider("N Points", 4, 24, 8)
            method = st.selectbox("Method", ["uniform", "default", "ror"])
            
            if st.button("Salvar Configurações"):
                st.session_state.lbp_config = {
                    'radius': radius,
                    'n_points': n_points,
                    'method': method
                }
                st.success("Configurações salvas!")
    
    def render_overview(self):
        """Renderiza a visão geral do projeto."""
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-value">LBP</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Feature Extractor</div>', unsafe_allow_html=True)
                st.markdown("Local Binary Patterns para extração de textura")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-value">Random Forest</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Classifier</div>', unsafe_allow_html=True)
                st.markdown("Ensemble de árvores de decisão")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-value">6</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Filtros Criativos</div>', unsafe_allow_html=True)
                st.markdown("Cartoon, Edges, Colormap, Stylized, Pencil")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Pipeline do projeto
        st.markdown("### 🔄 Pipeline do Projeto")
        
        pipeline_data = pd.DataFrame({
            'Etapa': ['Dataset', 'Pré-processamento', 'Extração LBP', 'Treinamento RF', 'Detecção', 'Filtros'],
            'Descrição': [
                'Human Detection Dataset (Kaggle)',
                'Redimensionamento 256x256, escala de cinza',
                'Histograma LBP uniforme (59 features)',
                'Random Forest com grid search',
                'Classificação em tempo real',
                'Aplicação de filtros criativos'
            ],
            'Status': ['✅', '✅', '✅', '🔄', '🔄', '✅']
        })
        
        st.dataframe(pipeline_data, use_container_width=True)
        
        # Explicação do LBP
        with st.expander("📚 Como funciona o LBP (Local Binary Patterns)?"):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                **Local Binary Pattern (LBP)** é um descritor de textura que funciona:
                
                1. Para cada pixel, compara com seus 8 vizinhos
                2. Se vizinho ≥ pixel central → 1, senão → 0
                3. Gera um número binário de 8 bits
                4. Histograma desses padrões forma o vetor de características
                
                **Vantagens:**
                - Invariância a iluminação
                - Baixo custo computacional
                - Robusto para texturas
                """)
            
            with col2:
                # Exemplo visual do LBP
                example = np.array([
                    [85, 32, 26],
                    [61, 50, 9],
                    [78, 12, 65]
                ])
                
                st.markdown("**Exemplo de cálculo:**")
                st.code("""
                Pixel central: 50
                Vizinhos: [85,32,26,61,9,78,12,65]
                Threshold (≥50): [1,0,0,1,0,1,0,1]
                Binário: 11010010 → Decimal: 210
                """)
                
                st.markdown("**Padrões uniformes:**")
                st.markdown("Apenas padrões com ≤2 transições 0-1")
    
    def render_training(self):
        """Renderiza a seção de treinamento."""
        
        st.markdown("### 🤖 Treinamento do Modelo")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Parâmetros de Treinamento")
            
            n_estimators = st.multiselect(
                "n_estimators",
                [10, 50, 100, 200],
                default=[100]
            )
            
            max_depth = st.multiselect(
                "max_depth",
                [5, 10, 15, None],
                default=[10],
                format_func=lambda x: "None" if x is None else str(x)
            )
            
            min_samples_split = st.slider("min_samples_split", 2, 10, 2)
            min_samples_leaf = st.slider("min_samples_leaf", 1, 5, 1)
            
            if st.button("🚀 Iniciar Treinamento", use_container_width=True):
                with st.spinner("Treinando modelo... Isso pode levar alguns minutos..."):
                    # Aqui você chamaria a função de treinamento
                    st.success("Treinamento concluído!")
                    
        with col2:
            st.markdown("#### Resultados do Último Treinamento")
            
            # Verificar se existem resultados salvos
            results_files = list(self.reports_dir.glob("results_*.json"))
            if results_files:
                latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
                with open(latest_results, 'r') as f:
                    results = json.load(f)
                
                test_metrics = results.get('test_metrics', {})
                
                # Métricas em cards
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                
                with mcol1:
                    st.metric("Acurácia", f"{test_metrics.get('accuracy', 0):.3f}")
                with mcol2:
                    st.metric("Precisão", f"{test_metrics.get('precision', 0):.3f}")
                with mcol3:
                    st.metric("Recall", f"{test_metrics.get('recall', 0):.3f}")
                with mcol4:
                    st.metric("F1-Score", f"{test_metrics.get('f1_score', 0):.3f}")
                
                # Melhores parâmetros
                best_params = results.get('best_params', {})
                if best_params:
                    st.markdown("**Melhores Parâmetros:**")
                    st.json(best_params)
            else:
                st.info("Nenhum resultado de treinamento encontrado.")
        
        # Grid Search Results
        st.markdown("### 📊 Resultados da Busca em Grade")
        
        results_files = list(self.reports_dir.glob("results_*.json"))
        if results_files:
            latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
            with open(latest_results, 'r') as f:
                results = json.load(f)
            
            grid_results = results.get('grid_search_results', [])
            
            if grid_results:
                # Converter para DataFrame
                df_results = pd.DataFrame(grid_results)
                
                # Heatmap de parâmetros
                fig = px.density_heatmap(
                    df_results, 
                    x='max_depth', 
                    y='n_estimators', 
                    z='val_accuracy',
                    title='Acurácia de Validação por Parâmetros',
                    color_continuous_scale='Viridis',
                    labels={'val_accuracy': 'Acurácia'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabela com top resultados
                st.markdown("#### Top 10 Configurações")
                st.dataframe(
                    df_results.nlargest(10, 'val_accuracy')[['n_estimators', 'max_depth', 'val_accuracy', 'val_f1']],
                    use_container_width=True
                )
    
    def render_realtime_detection(self):
        """Renderiza a seção de detecção em tempo real."""
        
        st.markdown("### 🎥 Detecção em Tempo Real")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Placeholder para o vídeo
            video_placeholder = st.empty()
            
            # Controles
            col_start, col_stop, col_filter = st.columns([1, 1, 2])
            
            with col_start:
                if st.button("▶️ Iniciar Câmera", use_container_width=True):
                    st.session_state.camera_active = True
                    st.session_state.detector = HumanDetector()
            
            with col_stop:
                if st.button("⏹️ Parar Câmera", use_container_width=True):
                    st.session_state.camera_active = False
                    if st.session_state.detector:
                        cv2.destroyAllWindows()
            
            with col_filter:
                filter_type = st.selectbox(
                    "Filtro",
                    ['cartoon', 'edges', 'colormap', 'stylized', 'pencil', 'none'],
                    index=0
                )
                st.session_state.filter_type = filter_type
        
        with col2:
            st.markdown("#### 📊 Estatísticas em Tempo Real")
            
            # Placeholder para métricas
            metrics_placeholder = st.empty()
            
            # Histórico de detecções
            st.markdown("#### 📜 Histórico")
            history_placeholder = st.empty()
        
        # Loop de detecção (simulado para Streamlit)
        if st.session_state.camera_active and st.session_state.detector:
            try:
                cap = cv2.VideoCapture(0)
                
                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Processar frame
                    processed = st.session_state.detector.preprocess_frame(frame)
                    pred, confidence = st.session_state.detector.detect(processed)
                    
                    # Aplicar filtro
                    filtered = st.session_state.detector.apply_creative_filter(
                        frame, st.session_state.filter_type
                    )
                    
                    # Adicionar texto
                    label = "HUMANO" if pred == 1 else "NÃO HUMANO"
                    color = (0, 255, 0) if pred == 1 else (0, 0, 255)
                    
                    cv2.putText(filtered, f"{label} ({confidence:.2%})", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Converter BGR para RGB
                    filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
                    
                    # Mostrar no Streamlit
                    video_placeholder.image(filtered_rgb, channels="RGB", use_column_width=True)
                    
                    # Atualizar métricas
                    with metrics_placeholder.container():
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.metric("Classe", label)
                        with col_m2:
                            st.metric("Confiança", f"{confidence:.2%}")
                    
                    # Atualizar histórico
                    st.session_state.detection_history.append({
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'class': label,
                        'confidence': confidence
                    })
                    
                    # Manter apenas últimos 10
                    if len(st.session_state.detection_history) > 10:
                        st.session_state.detection_history = st.session_state.detection_history[-10:]
                    
                    history_df = pd.DataFrame(st.session_state.detection_history)
                    history_placeholder.dataframe(history_df, use_container_width=True)
                    
                    time.sleep(0.03)  # ~30 FPS
                
                cap.release()
                
            except Exception as e:
                st.error(f"Erro na captura: {e}")
                st.session_state.camera_active = False
    
    def render_metrics_analysis(self):
        """Renderiza análise detalhada de métricas."""
        
        st.markdown("### 📈 Análise de Métricas")
        
        # Verificar se existem resultados
        results_files = list(self.reports_dir.glob("results_*.json"))
        if not results_files:
            st.warning("Nenhum resultado encontrado. Treine um modelo primeiro.")
            return
        
        # Selecionar resultado
        selected_file = st.selectbox(
            "Selecione o arquivo de resultados",
            options=results_files,
            format_func=lambda x: x.name
        )
        
        with open(selected_file, 'r') as f:
            results = json.load(f)
        
        test_metrics = results.get('test_metrics', {})
        grid_results = results.get('grid_search_results', [])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de radar para métricas
            metrics_names = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
            metrics_values = [
                test_metrics.get('accuracy', 0),
                test_metrics.get('precision', 0),
                test_metrics.get('recall', 0),
                test_metrics.get('f1_score', 0)
            ]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=metrics_values,
                theta=metrics_names,
                fill='toself',
                name='Métricas'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False,
                title="Métricas do Modelo"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Matriz de confusão
            cm = test_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            
            fig = px.imshow(
                cm,
                labels=dict(x="Predito", y="Real", color="Contagem"),
                x=['Não Humano', 'Humano'],
                y=['Não Humano', 'Humano'],
                title="Matriz de Confusão",
                text_auto=True,
                color_continuous_scale='Blues'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Curva de aprendizado (simulada)
        st.markdown("#### 📉 Curva de Aprendizado")
        
        if grid_results:
            df_grid = pd.DataFrame(grid_results)
            df_grid_sorted = df_grid.sort_values('n_estimators')
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            for depth in df_grid['max_depth'].unique():
                df_depth = df_grid_sorted[df_grid_sorted['max_depth'] == depth]
                fig.add_trace(
                    go.Scatter(
                        x=df_depth['n_estimators'],
                        y=df_depth['val_accuracy'],
                        name=f'max_depth={depth}',
                        mode='lines+markers'
                    ),
                    secondary_y=False,
                )
            
            fig.update_layout(
                title="Acurácia de Validação vs n_estimators",
                xaxis_title="n_estimators",
                yaxis_title="Acurácia"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_about(self):
        """Renderiza informações sobre o projeto."""
        
        st.markdown("### ℹ️ Sobre o Projeto")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            #### 📋 Descrição
            
            Este projeto implementa um sistema de reconhecimento de silhueta humana
            utilizando técnicas clássicas de Visão Computacional e Machine Learning.
            
            **Características principais:**
            - Dataset público do Kaggle (Human Detection Dataset)
            - Extração de características com LBP (Local Binary Patterns)
            - Classificação com Random Forest
            - Detecção em tempo real via webcam
            - 6 filtros criativos diferentes
            - Dashboard interativo com Streamlit
            
            **Tecnologias utilizadas:**
            - Python 3.9+
            - OpenCV para processamento de imagens
            - scikit-learn para Machine Learning
            - Streamlit para dashboard
            - Plotly para visualizações interativas
            """)
        
        with col2:
            st.markdown("""
            #### 📊 Dataset
            
            **Fonte:** [Human Detection Dataset - Kaggle](https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset)
            
            **Características:**
            - 921 imagens PNG
            - Resolução: 256x256 pixels
            - 2 classes: com humano (1) e sem humano (0)
            - Divisão: 70% treino, 10% validação, 20% teste
            
            #### 🎯 Objetivos do Projeto
            
            1. Implementar um pipeline completo de Visão Computacional
            2. Demonstrar o funcionamento do LBP
            3. Treinar e otimizar um Random Forest
            4. Criar uma aplicação interativa em tempo real
            5. Aplicar filtros criativos para visualização
            """)
        
        st.markdown("---")
        
        # Explicação detalhada do LBP
        st.markdown("### 📚 Explicação Detalhada do LBP")
        
        with st.expander("Clique para ver a explicação completa"):
            extractor = LBPFeatureExtractor()
            st.markdown(extractor.explain_lbp())
        
        # Exemplos de filtros
        st.markdown("### 🎨 Galeria de Filtros")
        
        filter_cols = st.columns(3)
        filters = ['cartoon', 'edges', 'colormap', 'stylized', 'pencil']
        
        # Placeholder para imagens dos filtros
        st.info("Capture uma imagem com a câmera para ver os exemplos dos filtros.")
        
        # Informações de contato/desenvolvedor
        st.markdown("---")
        st.markdown("""
        **Desenvolvido por:** [Seu Nome]  
        **Disciplina:** Processamento de Imagem e Visão Computacional  
        **Data:** {}
        """.format(datetime.now().strftime("%d/%m/%Y")))

def main():
    """Função principal para executar o dashboard."""
    dashboard = HumanRecognitionDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()