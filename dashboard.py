#!/usr/bin/env python3
"""
Dashboard Streamlit para reconhecimento de silhueta humana.
5 Tabs: Métricas Gerais, Por Fold, Detecção, Análise Visual, Config
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import time
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from src.feature_extractor import LBPFeatureExtractor
from src.config import MODELS_DIR, TARGET_SIZE, YOOSEE_CONFIG
from src.yoosee_camera import YooseeCamera

st.set_page_config(
    page_title="Human Recognition Dashboard",
    page_icon="👤",
    layout="wide"
)

REPORTS_DIR = Path("reports")


def init_session_state():
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'camera_source' not in st.session_state:
        st.session_state.camera_source = 'webcam'
    if 'yoosee_camera' not in st.session_state:
        st.session_state.yoosee_camera = None
    if 'yoosee_connected' not in st.session_state:
        st.session_state.yoosee_connected = False
    if 'filter_type' not in st.session_state:
        st.session_state.filter_type = 'cartoon'
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    if 'selected_report' not in st.session_state:
        st.session_state.selected_report = None


init_session_state()


def load_reports() -> List[Path]:
    reports = list(REPORTS_DIR.glob("model_comparison_*.json"))
    return sorted(reports, key=lambda p: p.stat().st_mtime, reverse=True)


def load_report_data(report_path: Path) -> Dict:
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_latest_report() -> Optional[Path]:
    reports = load_reports()
    return reports[0] if reports else None


class HumanDetector:
    def __init__(self):
        self.model = None
        self.extractor = None
        self.load_model()
    
    def load_model(self):
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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, TARGET_SIZE)
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)
        return frame_gray
    
    def detect(self, frame):
        features = self.extractor.extract_from_image(frame)
        features = features.reshape(1, -1)
        pred = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0]
        confidence = np.max(proba)
        return pred, confidence
    
    def apply_filter(self, frame, filter_type):
        if filter_type == 'cartoon':
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


def render_tab_metrics_general():
    st.markdown("## 📊 Métricas Gerais dos Modelos")
    
    reports = load_reports()
    if not reports:
        st.warning("Nenhum relatório encontrado em `/reports`. Execute o treinamento primeiro.")
        return
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_report = st.selectbox(
            "Selecione o relatório:",
            options=reports,
            format_func=lambda p: f"{p.name} ({datetime.fromtimestamp(p.stat().st_mtime).strftime('%d/%m/%Y %H:%M')})",
            index=0
        )
    with col2:
        if st.button("🔄 Atualizar", use_container_width=True):
            st.rerun()
    
    data = load_report_data(selected_report)
    results = data.get("results", [])
    
    if not results:
        st.warning("Relatório vazio.")
        return
    
    metrics_data = []
    for r in results:
        name = r.get("model_name", "unknown")
        test_m = r.get("test_metrics", {})
        cv_m = r.get("cv_metrics", {})
        cv_std = r.get("cv_std", {})
        
        metrics_data.append({
            "Modelo": name.replace("_", " ").title(),
            "Accuracy (CV)": f"{cv_m.get('accuracy', 0):.4f} ± {cv_std.get('accuracy', 0):.4f}",
            "Accuracy (Test)": f"{test_m.get('accuracy', 0):.4f}",
            "Precision": f"{test_m.get('precision', 0):.4f}",
            "Recall": f"{test_m.get('recall', 0):.4f}",
            "F1-Score": f"{test_m.get('f1_score', 0):.4f}",
            "Tempo (s)": f"{test_m.get('training_time', 0):.2f}"
        })
    
    df = pd.DataFrame(metrics_data)
    
    st.markdown("### Resumo de Todos os Modelos")
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    ranking = data.get("ranking", {})
    if ranking.get("test_accuracy"):
        best_model = ranking["test_accuracy"][0]
        st.success(f"🏆 **Melhor modelo por Accuracy:** `{best_model}`")
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Baixar CSV", csv, "metrics_summary.csv", "text/csv")


def render_tab_metrics_fold():
    st.markdown("## 📈 Métricas por Fold")
    
    reports = load_reports()
    if not reports:
        st.warning("Nenhum relatório encontrado.")
        return
    
    selected_report = st.selectbox("Relatório:", reports, 
                                   format_func=lambda p: p.name, key="fold_report")
    data = load_report_data(selected_report)
    results = data.get("results", [])
    
    if not results:
        return
    
    model_names = [r.get("model_name", "unknown") for r in results]
    selected_model = st.selectbox("Selecione o Modelo:", model_names, key="fold_model")
    
    model_data = next((r for r in results if r.get("model_name") == selected_model), None)
    if not model_data:
        return
    
    fold_metrics = model_data.get("cv_fold_metrics", {})
    
    if not fold_metrics or not fold_metrics.get("accuracy"):
        st.warning("Métricas por fold não disponíveis. Treine novamente para capturar dados por fold.")
        
        cv_std = model_data.get("cv_std", {})
        if cv_std:
            st.info(f"**CV Accuracy:** {model_data.get('cv_metrics', {}).get('accuracy', 0):.4f} ± {cv_std.get('accuracy', 0):.4f}")
        return
    
    st.markdown(f"### Modelo: `{selected_model}`")
    
    n_folds = len(fold_metrics["accuracy"])
    fold_data = []
    for i in range(n_folds):
        fold_data.append({
            "Fold": i + 1,
            "Accuracy": fold_metrics["accuracy"][i] if i < len(fold_metrics["accuracy"]) else 0,
            "Precision": fold_metrics["precision"][i] if i < len(fold_metrics["precision"]) else 0,
            "Recall": fold_metrics["recall"][i] if i < len(fold_metrics["recall"]) else 0,
            "F1-Score": fold_metrics["f1_score"][i] if i < len(fold_metrics["f1_score"]) else 0,
        })
    
    df = pd.DataFrame(fold_data)
    
    avg_row = {
        "Fold": "**Média**",
        "Accuracy": np.mean(fold_metrics["accuracy"]),
        "Precision": np.mean(fold_metrics["precision"]),
        "Recall": np.mean(fold_metrics["recall"]),
        "F1-Score": np.mean(fold_metrics["f1_score"]),
    }
    
    std_row = {
        "Fold": "**Std**",
        "Accuracy": np.std(fold_metrics["accuracy"]),
        "Precision": np.std(fold_metrics["precision"]),
        "Recall": np.std(fold_metrics["recall"]),
        "F1-Score": np.std(fold_metrics["f1_score"]),
    }
    
    df_summary = pd.DataFrame([avg_row, std_row])
    df_full = pd.concat([df, df_summary], ignore_index=True)
    
    st.dataframe(df_full.style.format({
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1-Score": "{:.4f}"
    }), use_container_width=True, hide_index=True)
    
    st.markdown("### 📊 Variação por Fold")
    chart_data = pd.DataFrame({
        'Fold': range(1, n_folds + 1),
        'Accuracy': fold_metrics["accuracy"],
        'F1-Score': fold_metrics["f1_score"]
    })
    st.bar_chart(chart_data.set_index('Fold'))


def render_tab_detection():
    st.markdown("## 🎥 Detecção em Tempo Real")
    
    if st.session_state.detector is None:
        if st.button("📦 Carregar Modelo", type="primary"):
            detector = HumanDetector()
            if detector.model:
                st.session_state.detector = detector
                st.success("Modelo carregado!")
                st.rerun()
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        video_placeholder = st.empty()
    
    with col2:
        st.markdown("### 🎨 Filtro")
        filter_type = st.selectbox(
            "Filtro:",
            ['none', 'cartoon', 'edges', 'colormap', 'stylized', 'pencil'],
            key="detect_filter"
        )
        st.session_state.filter_type = filter_type
        
        st.markdown("### 📊 Status")
        metrics_placeholder = st.empty()
        
        st.markdown("### 📜 Histórico")
        history_placeholder = st.empty()
    
    st.markdown("### 🎯 Fonte de Vídeo")
    col_webcam, col_yoosee = st.columns(2)
    
    with col_webcam:
        if st.button("📷 Webcam", use_container_width=True, disabled=st.session_state.camera_active):
            st.session_state.camera_source = 'webcam'
            st.session_state.camera_active = True
    
    with col_yoosee:
        if st.button("📹 Yoosee", use_container_width=True, disabled=st.session_state.camera_active):
            st.session_state.camera_source = 'yoosee'
            if not st.session_state.yoosee_connected:
                st.warning("Configure a Yoosee na tab Config primeiro!")
            else:
                st.session_state.camera_active = True
    
    if st.session_state.camera_active:
        if st.button("⏹️ Parar", type="secondary"):
            st.session_state.camera_active = False
            st.rerun()
        
        run_detection(video_placeholder, metrics_placeholder, history_placeholder)


def run_detection(video_placeholder, metrics_placeholder, history_placeholder):
    detector = st.session_state.detector
    cap = None
    
    try:
        if st.session_state.camera_source == 'yoosee':
            yoosee = st.session_state.yoosee_camera
            if not yoosee:
                st.error("Câmera não conectada")
                return
            source = "YOOSEE"
        else:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Webcam não disponível")
                return
            source = "WEBCAM"
        
        frame_count = 0
        start = time.time()
        
        while st.session_state.camera_active:
            if st.session_state.camera_source == 'yoosee':
                ret, frame = True, st.session_state.yoosee_camera.get_frame()
            else:
                ret, frame = cap.read()
            
            if not ret or frame is None:
                break
            
            processed = detector.preprocess_frame(frame)
            pred, conf = detector.detect(processed)
            filtered = detector.apply_filter(frame, st.session_state.filter_type)
            
            label = "HUMANO" if pred == 1 else "NÃO HUMANO"
            color = (0, 255, 0) if pred == 1 else (0, 0, 255)
            
            cv2.putText(filtered, f"[{source}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(filtered, f"{label} ({conf:.1%})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            frame_count += 1
            fps = frame_count / (time.time() - start) if frame_count > 0 else 0
            cv2.putText(filtered, f"FPS: {fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            video_placeholder.image(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB), channels="RGB")
            
            with metrics_placeholder.container():
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Classe", label)
                m_col2.metric("Confiança", f"{conf:.1%}")
            
            st.session_state.detection_history.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'class': label,
                'conf': f"{conf:.2%}"
            })
            
            if len(st.session_state.detection_history) > 10:
                st.session_state.detection_history = st.session_state.detection_history[-10:]
            
            history_placeholder.dataframe(pd.DataFrame(st.session_state.detection_history))
            
            time.sleep(0.03)
    
    except Exception as e:
        st.error(f"Erro: {e}")
    finally:
        if cap:
            cap.release()
        st.session_state.camera_active = False


def render_tab_analysis():
    st.markdown("## 📉 Análise Visual")
    
    reports = load_reports()
    if not reports:
        st.warning("Nenhum relatório encontrado.")
        return
    
    data = load_report_data(reports[0])
    results = data.get("results", [])
    
    if not results:
        return
    
    st.markdown("### 📊 Comparação de Acurácia")
    acc_data = {
        r.get("model_name", "unknown"): r.get("test_metrics", {}).get("accuracy", 0)
        for r in results
    }
    st.bar_chart(acc_data)
    
    st.markdown("### 📈 F1-Score por Modelo")
    f1_data = {
        r.get("model_name", "unknown"): r.get("test_metrics", {}).get("f1_score", 0)
        for r in results
    }
    st.bar_chart(f1_data)
    
    st.markdown("### 🏆 Ranking")
    ranking = data.get("ranking", {})
    for metric, models in ranking.items():
        st.markdown(f"**{metric}:**")
        for i, m in enumerate(models, 1):
            st.markdown(f"  {i}. `{m}`")


def render_tab_config():
    st.markdown("## ⚙️ Configurações")
    
    st.markdown("### 📹 Câmera Yoosee")
    
    if st.session_state.yoosee_connected:
        st.success(f"✅ Conectado: {YOOSEE_CONFIG.get('ip', 'N/A')}")
        if st.button("🔌 Desconectar"):
            st.session_state.yoosee_camera = None
            st.session_state.yoosee_connected = False
            st.rerun()
    else:
        with st.form("yoosee_config"):
            ip = st.text_input("IP", value=YOOSEE_CONFIG.get("ip", ""))
            user = st.text_input("Usuário", value=YOOSEE_CONFIG.get("username", "admin"))
            pwd = st.text_input("Senha", type="password", value=YOOSEE_CONFIG.get("password", ""))
            stream = st.selectbox("Stream", ["onvif1", "onvif2", "live"], index=0)
            
            if st.form_submit_button("🔌 Conectar"):
                try:
                    cam = YooseeCamera(ip=ip, username=user, password=pwd, stream_type=stream)
                    if cam.connect():
                        st.session_state.yoosee_camera = cam
                        st.session_state.yoosee_connected = True
                        st.success("Conectado!")
                        st.rerun()
                    else:
                        st.error("Falha na conexão")
                except Exception as e:
                    st.error(f"Erro: {e}")
    
    st.markdown("---")
    st.markdown("### 📂 Dataset")
    data_dir = Path("data/raw")
    if data_dir.exists():
        human = len(list((data_dir / "human").glob("*.png"))) if (data_dir / "human").exists() else 0
        no_human = len(list((data_dir / "no_human").glob("*.png"))) if (data_dir / "no_human").exists() else 0
        col1, col2 = st.columns(2)
        col1.metric("Humanos", human)
        col2.metric("Não Humanos", no_human)
    
    st.markdown("---")
    st.markdown("### ℹ️ Sobre")
    st.markdown("""
    **Human Recognition Project**
    
    Sistema de Visão Computacional para reconhecimento de silhueta humana.
    
    - **Features:** LBP (Local Binary Patterns)
    - **Modelos:** Random Forest, XGBoost, SVM, KNN, etc.
    - **Validação:** 5-fold Cross-Validation
    """)


def main():
    st.title("👤 Human Recognition Dashboard")
    st.markdown("Sistema de reconhecimento de silhueta humana")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Métricas Gerais",
        "📈 Métricas por Fold",
        "🎥 Detecção em Tempo Real",
        "📉 Análise Visual",
        "⚙️ Config/Sobre"
    ])
    
    with tab1:
        render_tab_metrics_general()
    
    with tab2:
        render_tab_metrics_fold()
    
    with tab3:
        render_tab_detection()
    
    with tab4:
        render_tab_analysis()
    
    with tab5:
        render_tab_config()


if __name__ == "__main__":
    main()
