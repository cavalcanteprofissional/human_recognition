#!/usr/bin/env python3
"""
Dashboard Streamlit para reconhecimento de silhueta humana.
Suporta webcam local e câmera Yoosee via RTSP.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import joblib
from pathlib import Path
from datetime import datetime

from src.feature_extractor import LBPFeatureExtractor
from src.config import MODELS_DIR, TARGET_SIZE, YOOSEE_CONFIG, get_yoosee_rtsp_url
from src.yoosee_camera import YooseeCamera

st.set_page_config(
    page_title="Human Recognition",
    page_icon="👤",
    layout="wide"
)

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


class HumanDetectorDashboard:
    """Dashboard para reconhecimento de silhueta humana."""
    
    def __init__(self):
        self.load_model()
    
    def load_model(self):
        """Carrega o modelo treinado."""
        models = list(MODELS_DIR.glob("model_*.pkl"))
        
        if not models:
            st.error("Nenhum modelo encontrado. Execute o treinamento primeiro.")
            return
        
        model_path = max(models, key=lambda p: p.stat().st_mtime)
        
        try:
            self.model = joblib.load(model_path)
            self.extractor = LBPFeatureExtractor(radius=1, n_points=8, method='uniform')
            st.session_state.detector = self
            st.success(f"Modelo carregado: {model_path.name}")
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {e}")
    
    def preprocess_frame(self, frame):
        """Pré-processa o frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, TARGET_SIZE)
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)
        return frame_gray
    
    def detect(self, frame):
        """Detecta humano no frame."""
        features = self.extractor.extract_from_image(frame)
        features = features.reshape(1, -1)
        pred = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0]
        confidence = np.max(proba)
        return pred, confidence
    
    def apply_filter(self, frame, filter_type):
        """Aplica filtro criativo."""
        if filter_type == 'cartoon':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255, 
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 9)
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
        else:
            return frame


def render_sidebar():
    """Renderiza a barra lateral."""
    st.image("https://raw.githubusercontent.com/opencv/opencv/master/doc/opencv-logo.png", 
             width=180)
    st.markdown("---")
    
    st.markdown("### 🎯 Fonte de Vídeo")
    
    camera_source = st.radio(
        "Selecione a câmera:",
        options=['webcam', 'yoosee'],
        format_func=lambda x: "📷 Webcam Local" if x == 'webcam' else "📹 Câmera Yoosee (IP)",
        horizontal=True,
        index=0 if st.session_state.camera_source == 'webcam' else 1
    )
    st.session_state.camera_source = camera_source
    
    if camera_source == 'yoosee':
        render_yoosee_sidebar()
    
    st.markdown("---")
    render_model_info()


def render_yoosee_sidebar():
    """Renderiza controles da câmera Yoosee."""
    st.markdown("### 📹 Câmera Yoosee")
    
    if st.session_state.yoosee_connected and st.session_state.yoosee_camera:
        st.success(f"✅ Conectado: {YOOSEE_CONFIG['ip']}")
        
        if st.button("🔌 Desconectar", use_container_width=True):
            st.session_state.yoosee_camera.disconnect()
            st.session_state.yoosee_camera = None
            st.session_state.yoosee_connected = False
            st.session_state.camera_source = 'webcam'
            st.rerun()
    else:
        st.info("Câmera desconectada")
        
        with st.form("yoosee_form"):
            ip = st.text_input("IP da Câmera", value=YOOSEE_CONFIG.get("ip", ""))
            username = st.text_input("Usuário", value=YOOSEE_CONFIG.get("username", "admin"))
            password = st.text_input("Senha", type="password", 
                                    value=YOOSEE_CONFIG.get("password", ""))
            stream = st.selectbox(
                "Stream",
                ["onvif1", "onvif2", "live", "stream11", "h264"],
                index=0
            )
            
            if st.form_submit_button("🔌 Conectar", use_container_width=True):
                try:
                    camera = YooseeCamera(
                        ip=ip,
                        username=username,
                        password=password,
                        stream_type=stream
                    )
                    
                    if camera.connect():
                        st.session_state.yoosee_camera = camera
                        st.session_state.yoosee_connected = True
                        st.success("Conectado!")
                        st.rerun()
                    else:
                        st.error("Falha ao conectar. Verifique as credenciais.")
                except Exception as e:
                    st.error(f"Erro: {e}")
        
        with st.expander("🔧 Testar Streams"):
            if st.button("Testar Todos os Streams"):
                if st.session_state.yoosee_camera:
                    with st.spinner("Testando..."):
                        results = st.session_state.yoosee_camera.test_streams()
                        for name, result in results.items():
                            status = "✅" if result.get("success") else "❌"
                            st.write(f"{status} {name}")


def render_model_info():
    """Renderiza informações do modelo."""
    st.markdown("### 📊 Status do Dataset")
    
    data_dir = Path("data")
    if data_dir.exists():
        raw_dir = data_dir / "raw"
        if raw_dir.exists():
            human_dir = raw_dir / "human"
            no_human_dir = raw_dir / "no_human"
            
            human_count = len(list(human_dir.glob("*.png"))) if human_dir.exists() else 0
            no_human_count = len(list(no_human_dir.glob("*.png"))) if no_human_dir.exists() else 0
            
            col1, col2 = st.columns(2)
            col1.metric("Humanos", human_count)
            col2.metric("Não Humanos", no_human_count)
    
    models = list(MODELS_DIR.glob("model_*.pkl"))
    if models:
        latest = max(models, key=lambda p: p.stat().st_mtime)
        st.caption(f"Modelo: {latest.name}")


def render_realtime_detection():
    """Renderiza a seção de detecção em tempo real."""
    st.markdown("### 🎥 Detecção em Tempo Real")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        video_placeholder = st.empty()
    
    with col2:
        st.markdown("#### 🎨 Filtro")
        filter_type = st.selectbox(
            "Selecione o filtro:",
            ['cartoon', 'edges', 'colormap', 'stylized', 'pencil', 'none'],
            index=['cartoon', 'edges', 'colormap', 'stylized', 'pencil', 'none'].index(
                st.session_state.filter_type
            )
        )
        st.session_state.filter_type = filter_type
        
        st.markdown("#### 📊 Estatísticas")
        metrics_placeholder = st.empty()
        
        st.markdown("#### 📜 Histórico")
        history_placeholder = st.empty()
    
    col_start, col_stop = st.columns(2)
    
    with col_start:
        if st.button("▶️ Iniciar", use_container_width=True, 
                     disabled=st.session_state.camera_active):
            if st.session_state.camera_source == 'yoosee' and not st.session_state.yoosee_connected:
                st.error("Conecte a câmera Yoosee primeiro!")
            else:
                st.session_state.camera_active = True
                st.session_state.detection_history = []
    
    with col_stop:
        if st.button("⏹️ Parar", use_container_width=True,
                    disabled=not st.session_state.camera_active):
            st.session_state.camera_active = False
    
    if st.session_state.camera_active:
        run_detection_loop(video_placeholder, metrics_placeholder, history_placeholder)


def run_detection_loop(video_placeholder, metrics_placeholder, history_placeholder):
    """Executa o loop de detecção."""
    detector = st.session_state.detector
    
    if not detector:
        st.error("Modelo não carregado!")
        return
    
    cap = None
    
    try:
        if st.session_state.camera_source == 'yoosee':
            yoosee_cam = st.session_state.yoosee_camera
            if not yoosee_cam:
                st.error("Câmera Yoosee não conectada")
                return
            yoosee_cam.start_streaming()
            source_label = "YOOSEE"
        else:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Erro ao abrir webcam")
                return
            source_label = "WEBCAM"
        
        frame_count = 0
        start_time = time.time()
        
        while st.session_state.camera_active:
            if st.session_state.camera_source == 'yoosee':
                frame = yoosee_cam.get_frame()
                if frame is None:
                    ret, frame = yoosee_cam.read_frame()
                else:
                    ret = frame is not None
            else:
                ret, frame = cap.read()
            
            if not ret or frame is None:
                break
            
            processed = detector.preprocess_frame(frame)
            pred, confidence = detector.detect(processed)
            
            filtered = detector.apply_filter(frame, st.session_state.filter_type)
            
            label = "HUMANO" if pred == 1 else "NÃO HUMANO"
            color = (0, 255, 0) if pred == 1 else (0, 0, 255)
            
            cv2.putText(filtered, f"[{source_label}]", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(filtered, f"{label} ({confidence:.1%})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(filtered, f"FPS: {fps:.1f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
            video_placeholder.image(filtered_rgb, channels="RGB", use_container_width=True)
            
            with metrics_placeholder.container():
                col1, col2 = st.columns(2)
                col1.metric("Classe", label)
                col2.metric("Confiança", f"{confidence:.1%}")
            
            st.session_state.detection_history.append({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'class': label,
                'confidence': confidence,
                'source': source_label
            })
            
            if len(st.session_state.detection_history) > 10:
                st.session_state.detection_history = st.session_state.detection_history[-10:]
            
            history_df = pd.DataFrame(st.session_state.detection_history)
            history_placeholder.dataframe(history_df, use_container_width=True)
            
            time.sleep(0.03)
        
        if cap:
            cap.release()
            
    except Exception as e:
        st.error(f"Erro: {e}")
    finally:
        st.session_state.camera_active = False


def render_about():
    """Renderiza a seção Sobre."""
    st.markdown("""
    ## 👤 Human Recognition Project
    
    Projeto de Visão Computacional para reconhecimento de silhueta humana em tempo real.
    
    ### 🔧 Tecnologias
    - **OpenCV**: Processamento de imagem
    - **LBP**: Extração de características
    - **Random Forest**: Classificação
    
    ### 📹 Fontes de Vídeo
    - Webcam local
    - Câmera IP Yoosee via RTSP
    
    ### 🎨 Filtros Disponíveis
    - Cartoon, Edges, Colormap, Stylized, Pencil
    """)


def main():
    """Função principal do dashboard."""
    st.title("👤 Human Recognition")
    st.markdown("Reconhecimento de silhueta humana em tempo real")
    
    tab1, tab2, tab3 = st.tabs(["🎥 Detecção", "ℹ️ Sobre", "🔧 Config"])
    
    with tab1:
        render_realtime_detection()
    
    with tab2:
        render_about()
    
    with tab3:
        render_sidebar()
    
    with st.sidebar:
        render_sidebar()


if __name__ == "__main__":
    main()
