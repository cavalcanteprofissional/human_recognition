import cv2
import numpy as np
import joblib
from pathlib import Path
import logging
from typing import Optional, Tuple
import time

from src.feature_extractor import LBPFeatureExtractor
from src.config import MODELS_DIR, TARGET_SIZE
from src.yoosee_camera import YooseeCamera

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HumanDetector:
    """
    Detector de humanos em tempo real usando webcam.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Inicializa o detector.
        
        Args:
            model_path: Caminho para o modelo treinado
        """
        # Carregar modelo
        if model_path is None:
            # Usar o modelo mais recente
            models = list(MODELS_DIR.glob("model_*.pkl"))
            if not models:
                raise FileNotFoundError("Nenhum modelo encontrado. Treine um modelo primeiro.")
            model_path = max(models, key=lambda p: p.stat().st_mtime)
        
        logger.info(f"Carregando modelo: {model_path}")
        self.model = joblib.load(model_path)
        
        # Inicializar extrator LBP
        self.extractor = LBPFeatureExtractor(
            radius=1,
            n_points=8,
            method='uniform'
        )
        
        # Configurações
        self.target_size = TARGET_SIZE
        self.confidence_threshold = 0.6
        
        # FPS calculation
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        
        # Proxy MJPEG
        self._mjpeg_streamer = None
        
        logger.info("Detector inicializado!")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Pré-processa o frame para extração de características.
        
        Args:
            frame: Frame da webcam (BGR)
            
        Returns:
            Frame processado (escala de cinza, redimensionado)
        """
        # Converter BGR para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Redimensionar
        frame_resized = cv2.resize(frame_rgb, self.target_size)
        
        # Converter para escala de cinza
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)
        
        return frame_gray
    
    def apply_creative_filter(self, frame: np.ndarray, 
                            filter_type: str = 'cartoon') -> np.ndarray:
        """
        Aplica filtros criativos ao frame.
        
        Args:
            frame: Frame original
            filter_type: Tipo de filtro ('cartoon', 'edges', 'colormap', 'stylized')
            
        Returns:
            Frame com filtro aplicado
        """
        if filter_type == 'cartoon':
            # Efeito cartoon
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255, 
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 9)
            
            color = cv2.bilateralFilter(frame, 9, 300, 300)
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            return cartoon
            
        elif filter_type == 'edges':
            # Detecção de bordas colorida
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return edges_color
            
        elif filter_type == 'colormap':
            # Mapas de cor criativos
            return cv2.applyColorMap(frame, cv2.COLORMAP_OCEAN)
            
        elif filter_type == 'stylized':
            # Efeito artístico
            return cv2.stylization(frame, sigma_s=60, sigma_r=0.6)
            
        elif filter_type == 'pencil':
            # Efeito de desenho a lápis
            gray, pencil = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
            return cv2.cvtColor(pencil, cv2.COLOR_GRAY2BGR)
            
        else:
            return frame
    
    def detect(self, frame: np.ndarray) -> Tuple[int, float]:
        """
        Detecta se há humano no frame.
        
        Args:
            frame: Frame pré-processado
            
        Returns:
            Tupla (classe_predita, confiança)
        """
        # Extrair características LBP
        features = self.extractor.extract_from_image(frame)
        
        # Reshape para o modelo
        features = features.reshape(1, -1)
        
        # Predizer
        pred = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0]
        confidence = np.max(proba)
        
        return pred, confidence
    
    def update_fps(self):
        """Atualiza o cálculo de FPS."""
        self.frame_count += 1
        if self.frame_count >= 10:
            current_time = time.time()
            self.fps = self.frame_count / (current_time - self.last_time)
            self.last_time = current_time
            self.frame_count = 0
    
    def run(self, filter_type: str = 'cartoon', source: str = 'webcam',
            yoosee_ip: str | None = None, yoosee_user: str | None = None, 
            yoosee_password: str | None = None, yoosee_stream: str = "onvif1",
            auto_find_yoosee: bool = False):
        """
        Executa o detector em tempo real.
        
        Args:
            filter_type: Tipo de filtro a aplicar
            source: Fonte de vídeo ('webcam' ou 'yoosee')
            yoosee_ip: IP da câmera Yoosee (se source='yoosee')
            yoosee_user: Usuário da câmera Yoosee
            yoosee_password: Senha da câmera Yoosee
            yoosee_stream: Stream da câmera ('onvif1', 'onvif2', etc.)
            auto_find_yoosee: Se True, busca IP automaticamente se conexão falhar
        """
        logger.info(f"Iniciando captura com filtro: {filter_type}, fonte: {source}")
        logger.info("Pressione 'q' para sair")
        logger.info("Pressione 'f' para trocar filtro")
        
        cap = None
        yoosee_cam = None
        
        if source == 'yoosee':
            from src.config import YOOSEE_CONFIG, find_and_update_yoosee_ip, reload_yoosee_config
            
            yoosee_ip = yoosee_ip or str(YOOSEE_CONFIG.get('ip', ''))
            yoosee_user = yoosee_user or str(YOOSEE_CONFIG.get('username', 'admin'))
            yoosee_password = yoosee_password or str(YOOSEE_CONFIG.get('password', ''))
            use_gateway = False
            http_url = None
            
            logger.info(f"Conectando à câmera Yoosee: {yoosee_ip}")
            yoosee_cam = YooseeCamera(
                ip=yoosee_ip,
                username=yoosee_user,
                password=yoosee_password,
                stream_type=yoosee_stream
            )
            
            if not yoosee_cam.connect():
                logger.warning("Conexão direta RTSP falhou")
                
                if auto_find_yoosee:
                    logger.info("Tentando encontrar câmera automaticamente na rede...")
                    ip, port, stream = find_and_update_yoosee_ip()
                    
                    if ip:
                        yoosee_ip = ip
                        yoosee_stream = stream or yoosee_stream
                        reload_yoosee_config()
                        
                        logger.info(f"Tentando conectar com novo IP: {yoosee_ip}")
                        yoosee_cam = YooseeCamera(
                            ip=yoosee_ip,
                            username=yoosee_user,
                            password=yoosee_password,
                            stream_type=yoosee_stream
                        )
                        
                        if not yoosee_cam.connect():
                            logger.warning("Tentando via gateway FFmpeg...")
                            use_gateway = True
                    else:
                        logger.warning("Câmera não encontrada na rede, tentando gateway FFmpeg...")
                        use_gateway = True
                else:
                    logger.warning("Tentando via gateway FFmpeg...")
                    use_gateway = True
                
                if use_gateway:
                    try:
                        from tools.rtsp_gateway import create_rtsp_url, start_rtsp_gateway, check_ffmpeg
                        
                        available, version = check_ffmpeg()
                        if available:
                            logger.info(f"FFmpeg: {version}")
                            
                            rtsp_url = create_rtsp_url(
                                yoosee_ip, 554, 
                                yoosee_user, yoosee_password, 
                                yoosee_stream
                            )
                            
                            success, result = start_rtsp_gateway(rtsp_url, http_port=8555)
                            
                            if success:
                                http_url = result
                                logger.info(f"Gateway FFmpeg ativo: {http_url}")
                            else:
                                logger.warning("Gateway FFmpeg falhou, tentando proxy PyAV...")
                                raise Exception("FFmpeg failed")
                        else:
                            logger.warning("FFmpeg não disponível, usando proxy PyAV...")
                            raise Exception("FFmpeg not available")
                    except Exception as e:
                        logger.info(f"Iniciando proxy PyAV (Digest Auth)...")
                        try:
                            import threading
                            from tools.rtsp_to_mjpeg import MJPEGStreamer, StreamConfig
                            
                            config = StreamConfig(
                                ip=yoosee_ip,
                                port=554,
                                username=yoosee_user,
                                password=yoosee_password,
                                stream_path=f"/{yoosee_stream}" if not yoosee_stream.startswith("/") else yoosee_stream,
                                local_port=8554
                            )
                            
                            mjpeg_streamer = MJPEGStreamer(config)
                            mjpeg_streamer.start()
                            time.sleep(2)
                            
                            http_url = f"http://localhost:8554/stream"
                            logger.info(f"Proxy PyAV ativo: {http_url}")
                            
                            self._mjpeg_streamer = mjpeg_streamer
                        except Exception as pe:
                            logger.error(f"Proxy PyAV também falhou: {pe}")
                            return
            
            if use_gateway and http_url:
                cap = cv2.VideoCapture(http_url)
                if not cap.isOpened():
                    logger.error("Não foi possível conectar via gateway")
                    return
                logger.info(f"Conectado via gateway HTTP: {http_url}")
                source_label = "YOOSEE (GW)"
            else:
                yoosee_cam.start_streaming()
                source_label = "YOOSEE"
        else:
            cap = cv2.VideoCapture(0)
            source_label = "WEBCAM"
            
            if not cap.isOpened():
                logger.error("Erro ao abrir webcam")
                return
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            source_label = "WEBCAM"
        
        filter_options = ['cartoon', 'edges', 'colormap', 'stylized', 'pencil', 'none']
        filter_idx = 0
        
        while True:
            # Ler frame da fonte apropriada
            if source == 'yoosee':
                if use_gateway and http_url:
                    ret, frame = cap.read()
                elif yoosee_cam:
                    frame = yoosee_cam.get_frame()
                    if frame is None:
                        ret, frame = yoosee_cam.read_frame()
                    else:
                        ret = frame is not None
                else:
                    ret, frame = False, None
            else:
                ret, frame = cap.read()
            
            if not ret or frame is None:
                if source == 'yoosee' and yoosee_cam and not use_gateway:
                    logger.warning("Tentando reconectar...")
                    if yoosee_cam.reconnect():
                        yoosee_cam.start_streaming()
                        continue
                break
            
            # Pré-processar para detecção
            processed_frame = self.preprocess_frame(frame)
            
            # Detectar
            pred, confidence = self.detect(processed_frame)
            
            # Aplicar filtro criativo
            current_filter = filter_options[filter_idx]
            if current_filter != 'none':
                filtered_frame = self.apply_creative_filter(frame, current_filter)
            else:
                filtered_frame = frame.copy()
            
            # Atualizar FPS
            self.update_fps()
            
            # Desenhar resultados
            label = "HUMANO" if pred == 1 else "NAO HUMANO"
            color = (0, 255, 0) if pred == 1 else (0, 0, 255)
            
            # Fundo semi-transparente para texto
            overlay = filtered_frame.copy()
            cv2.rectangle(overlay, (5, 5), (300, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, filtered_frame, 0.5, 0, filtered_frame)
            
            # Textos
            cv2.putText(filtered_frame, f"Classe: {label}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(filtered_frame, f"Confianca: {confidence:.2%}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(filtered_frame, f"FPS: {self.fps:.1f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(filtered_frame, f"Filtro: {current_filter}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Instruções
            cv2.putText(filtered_frame, "q: sair | f: trocar filtro", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Mostrar
            cv2.imshow('Human Recognition - Detector em Tempo Real', filtered_frame)
            
            # Teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                filter_idx = (filter_idx + 1) % len(filter_options)
                logger.info(f"Filtro alterado para: {filter_options[filter_idx]}")
        
        if source == 'yoosee':
            if use_gateway and http_url:
                if cap:
                    cap.release()
                try:
                    from tools.rtsp_gateway import stop_gateway
                    stop_gateway()
                    logger.info("Gateway FFmpeg parado")
                except:
                    pass
            elif yoosee_cam:
                yoosee_cam.disconnect()
            
            if self._mjpeg_streamer:
                self._mjpeg_streamer.stop()
                logger.info("Proxy PyAV parado")
        elif cap:
            cap.release()
        
        cv2.destroyAllWindows()
        logger.info("Detector encerrado.")

def main():
    """Função principal para execução do detector."""
    try:
        detector = HumanDetector()
        detector.run(filter_type='cartoon')
    except Exception as e:
        logger.error(f"Erro ao executar detector: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()