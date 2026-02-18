#!/usr/bin/env python3
"""
Módulo para conexão com câmeras Yoosee usando autenticação Digest.
O OpenCV não suporta autenticação Digest corretamente, então usamos um workaround.
"""

import cv2
import numpy as np
import logging
import hashlib
import socket
from typing import Optional, Tuple
from threading import Thread
from queue import Queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YooseeCameraDigest:
    """
    Câmera Yoosee com suporte a autenticação Digest.
    """
    
    def __init__(self, ip: str, username: str, password: str, 
                 port: int = 554, stream_type: str = "onvif1"):
        self.ip = ip
        self.username = username
        self.password = password
        self.port = port
        self.stream_type = stream_type
        
        self.rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/{stream_type}"
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.frame_queue = Queue(maxsize=30)
        self.thread: Optional[Thread] = None
        self.running = False
        
        # Para autenticação Digest
        self.realm: Optional[str] = None
        self.nonce: Optional[str] = None
        
    def _create_digest_auth(self, method: str, uri: str) -> str:
        """Cria cabeçalho de autenticação Digest."""
        if not self.realm or not self.nonce:
            return ""
        
        ha1 = hashlib.md5(f"{self.username}:{self.realm}:{self.password}".encode()).hexdigest()
        ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
        response = hashlib.md5(f"{ha1}:{self.nonce}:{ha2}".encode()).hexdigest()
        
        return (f'Digest username="{self.username}", realm="{self.realm}", '
                f'nonce="{self.nonce}", uri="{uri}", response="{response}"')
    
    def _get_digest_params(self) -> bool:
        """Obtém realm e nonce para autenticação Digest."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((self.ip, self.port))
            
            uri = f"rtsp://{self.ip}:{self.port}/{self.stream_type}"
            
            # OPTIONS
            request = f"OPTIONS {uri} RTSP/1.0\r\nCSeq: 1\r\n\r\n"
            sock.send(request.encode())
            sock.recv(4096)
            
            # DESCRIBE sem auth para obter nonce
            request = f"DESCRIBE {uri} RTSP/1.0\r\nCSeq: 2\r\n\r\n"
            sock.send(request.encode())
            response = sock.recv(4096).decode('utf-8', errors='replace')
            
            import re
            realm_match = re.search(r'realm="([^"]+)"', response)
            nonce_match = re.search(r'nonce="([^"]+)"', response)
            
            sock.close()
            
            if realm_match and nonce_match:
                self.realm = realm_match.group(1)
                self.nonce = nonce_match.group(1)
                logger.info(f"Digest auth: realm={self.realm}")
                return True
            
        except Exception as e:
            logger.error(f"Erro ao obter params Digest: {e}")
        
        return False
    
    def connect(self) -> bool:
        """Conecta à câmera."""
        logger.info(f"Conectando a {self.ip}...")
        
        # Primeiro, obter parâmetros de autenticação
        if not self._get_digest_params():
            logger.warning("Não foi possível obter params Digest, tentando conexão direta")
        
        # Tentar conexão com VideoCapture
        # O problema é que OpenCV não suporta Digest bem
        # Vamos tentar com a URL direta primeiro
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        if self.cap.isOpened():
            self.is_connected = True
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Conectado: {width}x{height} @ {fps}fps")
            return True
        
        # Se falhou, tentar com ffprobe como fallback
        self.cap = None
        return self._connect_with_ffmpeg()
    
    def _connect_with_ffmpeg(self) -> bool:
        """Conecta usando FFmpeg como bridge."""
        logger.info("Tentando conexão via FFmpeg...")
        
        # Verificar se FFmpeg está disponível
        import subprocess
        import os
        
        ffmpeg_paths = [
            r"C:\Users\muito\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe",
            "ffmpeg"
        ]
        
        ffmpeg_cmd = None
        for path in ffmpeg_paths:
            try:
                result = subprocess.run([path, "-version"], capture_output=True, timeout=5)
                if result.returncode == 0:
                    ffmpeg_cmd = path
                    break
            except:
                continue
        
        if not ffmpeg_cmd:
            logger.error("FFmpeg não encontrado")
            return False
        
        # Aqui teríamos que iniciar um servidor proxy RTSP
        # Por enquanto, retornar False
        logger.error("Fallback via FFmpeg não implementado completamente")
        return False
    
    def start_streaming(self):
        """Inicia thread de streaming."""
        if self.thread and self.thread.is_alive():
            return
        
        self.running = True
        self.thread = Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        logger.info("Thread de streaming iniciada")
    
    def _stream_loop(self):
        """Loop de streaming."""
        while self.running:
            ret, frame = self.read_frame()
            if ret and frame is not None:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                self.frame_queue.put(frame)
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Lê um frame."""
        if not self.is_connected or not self.cap:
            return False, None
        
        try:
            ret, frame = self.cap.read()
            return ret, frame
        except Exception as e:
            logger.error(f"Erro ao ler frame: {e}")
            return False, None
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Obtém frame da fila."""
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None
    
    def disconnect(self):
        """Desconecta."""
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.is_connected = False
        logger.info("Desconectado")


def create_camera(ip: str, username: str = "admin", password: str = "123",
                  port: int = 554, stream: str = "onvif1") -> YooseeCameraDigest:
    """Factory function."""
    return YooseeCameraDigest(ip, username, password, port, stream)


if __name__ == "__main__":
    import sys
    
    ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.100.47"
    user = sys.argv[2] if len(sys.argv) > 2 else "admin"
    pwd = sys.argv[3] if len(sys.argv) > 3 else "HonkaiImpact3rd"
    
    cam = create_camera(ip, user, pwd)
    
    if cam.connect():
        print("Conectado! Lendo 10 frames...")
        
        for i in range(10):
            ret, frame = cam.read_frame()
            if ret:
                print(f"Frame {i+1}: {frame.shape}")
            import time
            time.sleep(0.1)
        
        cam.disconnect()
    else:
        print("Falha ao conectar")
