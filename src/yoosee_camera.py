"""
Módulo para integração com câmeras Yoosee via RTSP.
Baseado em informações da comunidade e documentação disponível [citation:1][citation:2][citation:4]
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple, Generator
from threading import Thread, Lock
from queue import Queue
from dataclasses import dataclass
from pathlib import Path

from src.config import get_yoosee_rtsp_url, YOOSEE_RTSP_PATHS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CameraInfo:
    """Informações da câmera Yoosee."""
    ip: str
    port: int
    model: Optional[str] = None
    firmware: Optional[str] = None
    supported_streams: list = None

class YooseeCamera:
    """
    Classe para gerenciar conexão com câmera Yoosee via RTSP.
    
    As câmeras Yoosee utilizam protocolo RTSP/ONVIF para transmissão de vídeo [citation:2].
    Os endpoints mais comuns são /onvif1 (stream principal) e /onvif2 (sub-stream) [citation:3][citation:5].
    """
    
    def __init__(self, rtsp_url: str = None, username: str = None, 
                 password: str = None, ip: str = None, port: int = 554,
                 stream_type: str = "onvif1", auto_reconnect: bool = True):
        """
        Inicializa a câmera Yoosee.
        
        Args:
            rtsp_url: URL RTSP completa (se fornecida, ignora outros parâmetros)
            username: Nome de usuário
            password: Senha
            ip: Endereço IP da câmera
            port: Porta RTSP (padrão 554)
            stream_type: Tipo de stream ('onvif1', 'onvif2', 'live', 'stream11', 'h264')
            auto_reconnect: Tentar reconectar automaticamente em caso de falha
        """
        self.auto_reconnect = auto_reconnect
        self.is_connected = False
        self.cap = None
        self.frame_queue = Queue(maxsize=30)
        self.thread = None
        self.running = False
        self.lock = Lock()
        self.reconnect_count = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 2  # segundos
        
        # Construir URL RTSP [citation:4][citation:8]
        if rtsp_url:
            self.rtsp_url = rtsp_url
        elif ip:
            # Formato: rtsp://usuario:senha@ip:porta/caminho
            auth = f"{username}:{password}@" if username and password else ""
            path = YOOSEE_RTSP_PATHS.get(stream_type, "/onvif1")
            self.rtsp_url = f"rtsp://{auth}{ip}:{port}{path}"
        else:
            # Usar configurações padrão do .env
            self.rtsp_url = get_yoosee_rtsp_url(stream_type)
        
        logger.info(f"Câmera Yoosee inicializada com URL: {self._mask_password(self.rtsp_url)}")
    
    def _mask_password(self, url: str) -> str:
        """Mascara a senha na URL para logging."""
        if "@" in url:
            # Encontra parte entre :// e @
            prefix, rest = url.split("://", 1)
            auth, address = rest.split("@", 1)
            if ":" in auth:
                user, _ = auth.split(":", 1)
                return f"{prefix}://{user}:****@{address}"
        return url
    
    def connect(self) -> bool:
        """
        Estabelece conexão com a câmera.
        
        O OpenCV VideoCapture suporta URLs RTSP nativamente [citation:2][citation:8].
        
        Returns:
            True se conectou com sucesso
        """
        try:
            logger.info(f"Conectando à câmera Yoosee...")
            
            # Criar VideoCapture com buffer reduzido para menor latência
            self.cap = cv2.VideoCapture(self.rtsp_url)
            
            # Configurar parâmetros para melhor performance
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduz buffer
            self.cap.set(cv2.CAP_PROP_FPS, 15)        # Limita FPS
            
            # Aguardar conexão
            time.sleep(1)
            
            if self.cap.isOpened():
                self.is_connected = True
                self.reconnect_count = 0
                
                # Obter informações do stream
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                logger.info(f"✅ Câmera conectada: {width}x{height} @ {fps:.1f}fps")
                return True
            else:
                logger.error("❌ Falha ao conectar: VideoCapture não abriu")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erro ao conectar: {e}")
            return False
    
    def disconnect(self):
        """Desconecta a câmera e libera recursos."""
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.is_connected = False
        
        logger.info("Câmera desconectada")
    
    def reconnect(self) -> bool:
        """
        Tenta reconectar à câmera.
        
        Implementa lógica de reconexão para lidar com instabilidades de rede [citation:8].
        
        Returns:
            True se reconectou com sucesso
        """
        self.disconnect()
        
        while self.reconnect_count < self.max_reconnect_attempts:
            self.reconnect_count += 1
            logger.info(f"Tentativa de reconexão {self.reconnect_count}/{self.max_reconnect_attempts}")
            
            if self.connect():
                return True
            
            time.sleep(self.reconnect_delay * self.reconnect_count)  # Backoff exponencial
        
        logger.error("Número máximo de tentativas de reconexão atingido")
        return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lê um frame da câmera.
        
        Returns:
            Tupla (sucesso, frame)
        """
        if not self.is_connected or not self.cap:
            if self.auto_reconnect:
                if self.reconnect():
                    return self.cap.read() if self.cap else (False, None)
            return False, None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret and self.auto_reconnect:
                logger.warning("Frame perdido, tentando reconectar...")
                return self.reconnect_and_read()
            
            return ret, frame
            
        except Exception as e:
            logger.error(f"Erro ao ler frame: {e}")
            if self.auto_reconnect:
                return self.reconnect_and_read()
            return False, None
    
    def reconnect_and_read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Tenta reconectar e ler um frame."""
        if self.reconnect():
            return self.cap.read() if self.cap else (False, None)
        return False, None
    
    def start_streaming(self):
        """
        Inicia thread para streaming contínuo.
        Útil para reduzir latência em aplicações em tempo real.
        """
        if self.thread and self.thread.is_alive():
            return
        
        self.running = True
        self.thread = Thread(target=self._streaming_loop, daemon=True)
        self.thread.start()
        logger.info("Thread de streaming iniciada")
    
    def _streaming_loop(self):
        """Loop principal de streaming em thread separada."""
        while self.running:
            ret, frame = self.read_frame()
            if ret and frame is not None:
                if self.frame_queue.full():
                    # Remove frame mais antigo se fila cheia
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01)  # Evita loop excessivo em caso de erro
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Obtém o frame mais recente da fila de streaming.
        
        Returns:
            Frame mais recente ou None
        """
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None
    
    def get_info(self) -> CameraInfo:
        """Obtém informações da câmera."""
        return CameraInfo(
            ip=self.rtsp_url.split("@")[-1].split(":")[0] if "@" in self.rtsp_url else "unknown",
            port=554,
            supported_streams=list(YOOSEE_RTSP_PATHS.keys())
        )
    
    def test_streams(self) -> dict:
        """
        Testa todos os endpoints RTSP para encontrar o que funciona.
        
        Returns:
            Dicionário com resultados dos testes
        """
        results = {}
        
        for stream_name, path in YOOSEE_RTSP_PATHS.items():
            # Construir URL para este stream
            base_url = self.rtsp_url.rsplit("/", 1)[0]  # Remove caminho atual
            test_url = f"{base_url}{path}"
            
            logger.info(f"Testando stream: {stream_name} -> {path}")
            
            # Tentar conexão rápida
            cap = cv2.VideoCapture(test_url)
            time.sleep(0.5)
            
            if cap.isOpened():
                ret, frame = cap.read()
                results[stream_name] = {
                    "success": ret,
                    "url": test_url,
                    "path": path
                }
                if ret:
                    logger.info(f"  ✅ {stream_name} funcionou!")
                else:
                    logger.info(f"  ⚠️ {stream_name} conectou mas não retornou frames")
            else:
                results[stream_name] = {
                    "success": False,
                    "url": test_url,
                    "error": "Não conectou"
                }
                logger.info(f"  ❌ {stream_name} falhou")
            
            cap.release()
        
        return results

# Função de conveniência para criar câmera rapidamente
def create_yoosee_camera(ip: str = None, password: str = None, 
                         stream: str = "onvif1") -> YooseeCamera:
    """
    Cria e conecta uma câmera Yoosee.
    
    Args:
        ip: IP da câmera (usa .env se não fornecido)
        password: Senha (usa .env se não fornecido)
        stream: Tipo de stream
    
    Returns:
        Instância da câmera conectada
    """
    from src.config import YOOSEE_CONFIG
    
    camera = YooseeCamera(
        ip=ip or YOOSEE_CONFIG["ip"],
        username=YOOSEE_CONFIG["username"],
        password=password or YOOSEE_CONFIG["password"],
        stream_type=stream
    )
    
    if camera.connect():
        return camera
    else:
        raise ConnectionError("Não foi possível conectar à câmera Yoosee")

# Exemplo de uso
if __name__ == "__main__":
    # Teste rápido
    cam = YooseeCamera()
    
    if cam.connect():
        print("Câmera conectada! Testando streams...")
        results = cam.test_streams()
        
        print("\nResultados dos testes:")
        for name, result in results.items():
            status = "✅" if result.get("success") else "❌"
            print(f"{status} {name}: {result.get('url')}")
        
        # Mostrar alguns frames
        print("\nCapturando 10 frames...")
        for i in range(10):
            ret, frame = cam.read_frame()
            if ret:
                print(f"Frame {i+1}: {frame.shape}")
            time.sleep(0.1)
        
        cam.disconnect()
    else:
        print("Falha ao conectar. Verifique o IP e credenciais.")