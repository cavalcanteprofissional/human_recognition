#!/usr/bin/env python3
"""
Dashboard Streamlit para reconhecimento de silhueta humana.
6 Tabs: Métricas Gerais, Por Fold, Hiperparâmetros, Detecção, Análise Visual, Config
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.feature_extractor import LBPFeatureExtractor
from src.config import MODELS_DIR, TARGET_SIZE, YOOSEE_CONFIG

REPORTS_DIR = Path("reports")


class HumanDetector:
    def __init__(self):
        self.model = None
        self.extractor = None
        self.load_model()
    
    def load_model(self) -> bool:
        models = list(MODELS_DIR.glob("*.pkl"))
        if not models:
            return False
        
        best_models = [m for m in models if "best_model" in m.name]
        if best_models:
            model_path = max(best_models, key=lambda p: p.stat().st_mtime)
        else:
            model_path = max(models, key=lambda p: p.stat().st_mtime)
        
        try:
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict) and "model" in model_data:
                self.model = model_data["model"]
            else:
                self.model = model_data
            self.extractor = LBPFeatureExtractor(radius=1, n_points=8, method='uniform')
            return True
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {e}")
            return False
    
    def preprocess_frame(self, frame):
        if frame is None:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, TARGET_SIZE)
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)
        return frame_gray
    
    def detect(self, frame):
        if frame is None or self.model is None or self.extractor is None:
            return "ERRO", 0.0
        features = self.extractor.extract_from_image(frame)
        features = features.reshape(1, -1)
        pred = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0]
        confidence = np.max(proba)
        return "HUMANO" if pred == 1 else "NÃO HUMANO", confidence
    
    def apply_filter(self, frame, filter_type):
        if frame is None:
            return None
        if filter_type == 'none':
            return frame
        elif filter_type == 'cartoon':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(frame, 9, 300, 300)
            return cv2.bitwise_and(color, color, mask=edges)
        elif filter_type == 'edges':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif filter_type == 'colormap':
            return cv2.applyColorMap(frame, cv2.COLORMAP_OCEAN)
        elif filter_type == 'stylized':
            return cv2.stylization(frame, sigma_s=60, sigma_r=0.6)
        elif filter_type == 'pencil':
            _, pencil = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
            return cv2.cvtColor(pencil, cv2.COLOR_GRAY2BGR)
        return frame


@st.cache_data
def get_report_choices() -> List[str]:
    reports = list(REPORTS_DIR.glob("model_comparison_*.json"))
    return [r.name for r in sorted(reports, key=lambda p: p.stat().st_mtime, reverse=True)]


@st.cache_data
def load_report_data(report_name: str):
    report_path = REPORTS_DIR / report_name
    if not report_path.exists():
        return {}
    with open(report_path, 'r') as f:
        return json.load(f)


def get_model_choices(report_name: str) -> List[str]:
    data = load_report_data(report_name)
    results = data.get("results", [])
    return [r.get("model_name", "unknown") for r in results]


def render_tab_metrics_general(report_choice: str):
    if not report_choice:
        return pd.DataFrame(), "Selecione um relatório"
    
    data = load_report_data(report_choice)
    results = data.get("results", [])
    
    if not results:
        return pd.DataFrame(), "Relatório vazio"
    
    table_data = []
    for r in results:
        name = r.get("model_name", "unknown")
        cv_m = r.get("cv_metrics", {})
        test_m = r.get("test_metrics", {})
        cv_std = r.get("cv_std", {})
        
        table_data.append({
            "Modelo": name.replace("_", " ").title(),
            "Accuracy (CV)": f"{cv_m.get('accuracy', 0):.3f} ± {cv_std.get('accuracy', 0):.3f}",
            "Accuracy (Test)": f"{test_m.get('accuracy', 0):.3f}",
            "Precision": f"{test_m.get('precision', 0):.3f}",
            "Recall": f"{test_m.get('recall', 0):.3f}",
            "F1-Score": f"{test_m.get('f1_score', 0):.3f}",
            "Tempo (s)": f"{r.get('training_time', 0):.2f}"
        })
    
    df = pd.DataFrame(table_data)
    
    best_model = max(results, key=lambda r: r.get("test_metrics", {}).get("accuracy", 0))
    best_name = best_model.get("model_name", "unknown").replace("_", " ").title()
    best_acc = best_model.get("test_metrics", {}).get("accuracy", 0)
    best_text = f"**🏆 Melhor Modelo:** {best_name} (Accuracy: {best_acc:.1%})"
    
    return df, best_text


def render_tab_metrics_fold(report_choice: str, model_choice: str):
    if not report_choice or not model_choice:
        return pd.DataFrame(), None
    
    data = load_report_data(report_choice)
    results = data.get("results", [])
    
    model_data = next((r for r in results if r.get("model_name") == model_choice), None)
    if not model_data:
        return pd.DataFrame(), None
    
    fold_metrics = model_data.get("cv_fold_metrics", {})
    if not fold_metrics:
        return pd.DataFrame(), "Este relatório não contém métricas por fold"
    
    folds = len(fold_metrics.get("accuracy", []))
    table_data = []
    for i in range(folds):
        table_data.append({
            "Fold": i + 1,
            "Accuracy": fold_metrics.get("accuracy", [0])[i],
            "Precision": fold_metrics.get("precision", [0])[i],
            "Recall": fold_metrics.get("recall", [0])[i],
            "F1-Score": fold_metrics.get("f1_score", [0])[i]
        })
    
    cv_m = model_data.get("cv_metrics", {})
    cv_std = model_data.get("cv_std", {})
    table_data.append({
        "Fold": "Média",
        "Accuracy": cv_m.get("accuracy", 0),
        "Precision": cv_m.get("precision", 0),
        "Recall": cv_m.get("recall", 0),
        "F1-Score": cv_m.get("f1_score", 0)
    })
    table_data.append({
        "Fold": "Std",
        "Accuracy": cv_std.get("accuracy", 0),
        "Precision": cv_std.get("precision", 0),
        "Recall": cv_std.get("recall", 0),
        "F1-Score": cv_std.get("f1_score", 0)
    })
    
    df = pd.DataFrame(table_data)
    
    chart_df = pd.DataFrame({
        "Fold": list(range(1, folds + 1)),
        "Accuracy": fold_metrics.get("accuracy", [])
    })
    
    return df, chart_df


def render_tab_hyperparams(report_choice: str, model_choice: str):
    if not report_choice or not model_choice:
        return pd.DataFrame(), "{}", ""
    
    data = load_report_data(report_choice)
    results = data.get("results", [])
    
    model_data = next((r for r in results if r.get("model_name") == model_choice), None)
    if not model_data:
        return pd.DataFrame(), "{}", ""
    
    params = model_data.get("best_params", {})
    cv_m = model_data.get("cv_metrics", {})
    cv_std = model_data.get("cv_std", {})
    val_m = model_data.get("val_metrics", {})
    test_m = model_data.get("test_metrics", {})
    
    table_data = [{
        "Modelo": model_choice.replace("_", " ").title(),
        "Hiperparâmetros": json.dumps(params),
        "CV Acc": f"{cv_m.get('accuracy', 0):.3f} ± {cv_std.get('accuracy', 0):.3f}",
        "Val Acc": f"{val_m.get('accuracy', 0):.3f}",
        "Test Acc": f"{test_m.get('accuracy', 0):.3f}",
        "Test F1": f"{test_m.get('f1_score', 0):.3f}"
    }]
    
    df = pd.DataFrame(table_data)
    params_json = json.dumps(params, indent=2)
    csv = df.to_csv(index=False)
    
    return df, params_json, csv


def render_tab_analysis(report_choice: str):
    if not report_choice:
        return pd.DataFrame(), "Selecione um relatório"
    
    data = load_report_data(report_choice)
    results = data.get("results", [])
    
    if not results:
        return pd.DataFrame(), "Relatório vazio"
    
    chart_data = []
    for r in results:
        name = r.get("model_name", "unknown")
        test_m = r.get("test_metrics", {})
        chart_data.append({
            "Modelo": name.replace("_", " ").title(),
            "Accuracy": test_m.get("accuracy", 0),
            "F1-Score": test_m.get("f1_score", 0),
        })
    
    df = pd.DataFrame(chart_data)
    
    ranking = data.get("ranking", {})
    ranking_text = "## Ranking de Modelos\n\n"
    for metric, models in ranking.items():
        ranking_text += f"### {metric}\n"
        for i, m in enumerate(models, 1):
            ranking_text += f"{i}. {m.replace('_', ' ').title()}\n"
        ranking_text += "\n"
    
    return df, ranking_text


def get_dataset_stats() -> Tuple[int, int]:
    data_dir = Path("data/raw")
    human = len(list((data_dir / "human").glob("*.png"))) if (data_dir / "human").exists() else 0
    no_human = len(list((data_dir / "no_human").glob("*.png"))) if (data_dir / "no_human").exists() else 0
    return human, no_human


def detect_from_image(image, filter_type: str):
    if image is None:
        return None, "Nenhuma imagem", "0%"
    
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    processed = detector.preprocess_frame(frame)
    label, conf = detector.detect(processed)
    
    filtered = detector.apply_filter(frame, filter_type)
    
    color = (0, 255, 0) if label == "HUMANO" else (0, 0, 255)
    cv2.putText(filtered, f"{label} ({conf:.1%})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    result_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    
    return result_rgb, label, f"{conf:.1%}"


def main():
    st.set_page_config(
        page_title="Human Recognition Dashboard",
        page_icon="👤",
        layout="wide"
    )
    
    st.title("👤 Human Recognition Dashboard")
    st.markdown("Sistema de reconhecimento de silhueta humana usando LBP + Machine Learning")
    
    tabs = st.tabs([
        "📊 Métricas Gerais",
        "📈 Métricas por Fold",
        "🔧 Hiperparâmetros",
        "📤 Detecção de Imagem",
        "📉 Análise Visual",
        "⚙️ Config/Sobre"
    ])
    
    with tabs[0]:
        st.header("Métricas Gerais dos Modelos")
        
        report_choice = st.selectbox(
            "Selecione o relatório",
            get_report_choices(),
            index=0 if get_report_choices() else None
        )
        
        if report_choice:
            df, best_text = render_tab_metrics_general(report_choice)
            
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                st.markdown(best_text)
                
                csv = df.to_csv(index=False)
                st.download_button("📥 Baixar CSV", csv, "metrics.csv", "text/csv")
        else:
            st.warning("Nenhum relatório encontrado")
    
    with tabs[1]:
        st.header("Métricas por Fold")
        
        col1, col2 = st.columns(2)
        with col1:
            report_choice_2 = st.selectbox(
                "Selecione o relatório",
                get_report_choices(),
                index=0 if get_report_choices() else None,
                key="report_2"
            )
        
        with col2:
            if report_choice_2:
                model_choices = get_model_choices(report_choice_2)
                model_choice = st.selectbox(
                    "Selecione o Modelo",
                    model_choices,
                    index=0 if model_choices else None
                )
            else:
                model_choice = None
        
        if report_choice_2 and model_choice:
            df, chart_df = render_tab_metrics_fold(report_choice_2, model_choice)
            
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                
                if chart_df is not None and not chart_df.empty:
                    st.bar_chart(chart_df.set_index("Fold"))
        else:
            st.warning("Selecione um relatório e um modelo")
    
    with tabs[2]:
        st.header("Hiperparâmetros e Métricas")
        
        col1, col2 = st.columns(2)
        with col1:
            report_choice_3 = st.selectbox(
                "Selecione o relatório",
                get_report_choices(),
                index=0 if get_report_choices() else None,
                key="report_3"
            )
        
        with col2:
            if report_choice_3:
                model_choices_3 = get_model_choices(report_choice_3)
                model_choice_3 = st.selectbox(
                    "Selecione o Modelo",
                    model_choices_3,
                    index=0 if model_choices_3 else None,
                    key="model_3"
                )
            else:
                model_choice_3 = None
        
        if report_choice_3 and model_choice_3:
            df, params_json, csv = render_tab_hyperparams(report_choice_3, model_choice_3)
            
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                
                st.subheader("Hiperparâmetros")
                st.code(params_json, language="json")
                
                st.download_button("📥 Baixar CSV", csv, "hyperparams.csv", "text/csv")
        else:
            st.warning("Selecione um relatório e um modelo")
    
    with tabs[3]:
        st.header("Detecção de Imagem")
        
        if detector.model:
            st.success("Modelo carregado com sucesso!")
        else:
            st.warning("Nenhum modelo encontrado. Execute o treinamento primeiro.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Envie uma imagem",
                type=['png', 'jpg', 'jpeg']
            )
        
        with col2:
            filter_type = st.selectbox(
                "Filtro",
                ['none', 'cartoon', 'edges', 'colormap', 'stylized', 'pencil']
            )
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            result_image, label, conf = detect_from_image(image, filter_type)
            
            if result_image is not None:
                st.image(result_image, caption="Resultado", channels="RGB")
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric("Classe", label)
                with col_res2:
                    st.metric("Confiança", conf)
    
    with tabs[4]:
        st.header("Análise Visual")
        
        report_choice_5 = st.selectbox(
            "Selecione o relatório",
            get_report_choices(),
            index=0 if get_report_choices() else None,
            key="report_5"
        )
        
        if report_choice_5:
            df, ranking_text = render_tab_analysis(report_choice_5)
            
            if not df.empty:
                st.subheader("Comparação de Accuracy e F1-Score")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.bar_chart(df.set_index("Modelo")["Accuracy"])
                with col2:
                    st.bar_chart(df.set_index("Modelo")["F1-Score"])
                
                st.markdown(ranking_text)
        else:
            st.warning("Selecione um relatório")
    
    with tabs[5]:
        st.header("Configurações")
        
        st.subheader("Câmera Yoosee")
        yoosee_info = f"""
        - **IP:** {YOOSEE_CONFIG.get('ip', 'N/A')}
        - **Porta:** {YOOSEE_CONFIG.get('port', 554)}
        - **Usuário:** {YOOSEE_CONFIG.get('username', 'admin')}
        - **Stream:** {YOOSEE_CONFIG.get('stream', 'onvif1')}
        """
        st.markdown(yoosee_info)
        
        st.divider()
        
        st.subheader("Dataset")
        human, no_human = get_dataset_stats()
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.metric("Humanos", human)
        with col_h2:
            st.metric("Não Humanos", no_human)
        
        st.divider()
        
        st.subheader("Sobre")
        st.markdown("""
        **Human Recognition Project**
        
        Sistema de Visão Computacional para reconhecimento de silhueta humana.
        
        - **Features:** LBP (Local Binary Patterns)
        - **Modelos:** Random Forest, XGBoost, SVM, KNN, Logistic Regression, MLP, Gradient Boosting, LightGBM
        - **Validação:** 5-fold Cross-Validation
        - **Framework:** Streamlit + OpenCV + scikit-learn
        """)


if __name__ == "__main__":
    detector = HumanDetector()
    main()
