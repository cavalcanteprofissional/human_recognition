#!/usr/bin/env python3
"""
Proxy HTTP que converte stream RTSP (H.265/H.264) para MJPEG.
Permite que o OpenCV consuma o stream via HTTP.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2
import av
import threading
import time
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional
import numpy as np
from dataclasses import dataclass
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    ip: str
    port: int
    username: str
    password: str
    stream_path: str = "/onvif1"
    local_port: int = 8554
    quality: int = 85


class MJPEGStreamer:
    """Converte stream RTSP para MJPEG HTTP."""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.running = False
        self.container: Optional[av.container.InputContainer] = None
        self.current_frame: Optional[bytes] = None
        self.frame_lock = threading.Lock()
        self._rtsp_thread: Optional[threading.Thread] = None
        self._http_thread: Optional[threading.Thread] = None
        self._http_server = None
        self.fps = 0
        self.frame_count = 0
        self.width = 0
        self.height = 0
        
    @property
    def stream_url(self) -> str:
        return f"rtsp://{self.config.username}:{self.config.password}@{self.config.ip}:{self.config.port}{self.config.stream_path}"
    
    def _connect(self) -> bool:
        """Conecta ao stream RTSP."""
        try:
            logger.info(f"Conectando a: {self.stream_url}")
            self.container = av.open(self.stream_url, timeout=10)
            logger.info("Stream conectado!")
            return True
        except Exception as e:
            logger.error(f"Erro ao conectar: {e}")
            return False
    
    def _disconnect(self):
        """Desconecta do stream."""
        if self.container:
            try:
                self.container.close()
            except:
                pass
            self.container = None
    
    def _convert_frame(self, frame: av.VideoFrame) -> Optional[bytes]:
        """Converte frame para JPEG."""
        try:
            img = frame.to_ndarray(format='bgr24')
            
            target_width = 1280
            aspect = img.shape[1] / img.shape[0]
            target_height = int(target_width / aspect)
            img = cv2.resize(img, (target_width, target_height))
            
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.quality]
            ret, jpg = cv2.imencode('.jpg', img, encode_param)
            
            if ret:
                return jpg.tobytes()
        except Exception as e:
            logger.error(f"Erro ao converter frame: {e}")
        
        return None
    
    def _streaming_loop(self):
        """Loop principal de streaming."""
        logger.info("Thread de streaming RTSP iniciada")
        
        while self.running:
            try:
                if not self.container:
                    if not self._connect():
                        time.sleep(2)
                        continue
                
                for packet in self.container.demux(video=0):
                    if not self.running:
                        break
                    
                    for frame in packet.decode():
                        if not self.running:
                            break
                        
                        jpg_data = self._convert_frame(frame)
                        
                        if jpg_data:
                            with self.frame_lock:
                                self.current_frame = jpg_data
                                self.frame_count += 1
                                
                                if self.frame_count == 1:
                                    self.width = frame.width
                                    self.height = frame.height
                            
                            time.sleep(0.01)
                
            except av.error.InvalidDataError as e:
                logger.warning(f"Erro de dados, reconectando... {e}")
                self._disconnect()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Erro no streaming: {e}")
                self._disconnect()
                time.sleep(2)
        
        logger.info("Thread de streaming finalizada")
    
    def _run_http_server(self):
        """Executa o servidor HTTP."""
        MJPEGHandler.streamer = self
        try:
            self._http_server = ThreadedHTTPServer(('0.0.0.0', self.config.local_port), MJPEGHandler)
            logger.info(f"Servidor HTTP iniciado em http://0.0.0.0:{self.config.local_port}")
            self._http_server.serve_forever()
        except Exception as e:
            logger.error(f"Erro no servidor HTTP: {e}")
    
    def start(self):
        """Inicia o streamer."""
        if self.running:
            return
        
        self.running = True
        
        self._rtsp_thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self._rtsp_thread.start()
        
        self._http_thread = threading.Thread(target=self._run_http_server, daemon=True)
        self._http_thread.start()
        
        time.sleep(1)
        logger.info(f"Streamer iniciado na porta {self.config.local_port}")
    
    def stop(self):
        """Para o streamer."""
        self.running = False
        
        if self._http_server:
            try:
                self._http_server.shutdown()
            except:
                pass
        
        if self._rtsp_thread:
            self._rtsp_thread.join(timeout=5)
        
        if self._http_thread:
            self._http_thread.join(timeout=2)
        
        self._disconnect()
        logger.info("Streamer parado")
    
    def get_frame(self) -> Optional[bytes]:
        """Obtém o frame atual em JPEG."""
        with self.frame_lock:
            return self.current_frame


class MJPEGHandler(BaseHTTPRequestHandler):
    """Handler HTTP para stream MJPEG."""
    
    streamer: Optional[MJPEGStreamer] = None
    
    def log_message(self, format, *args):
        pass
    
    def do_GET(self):
        if self.path == '/stream' or self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            try:
                while self.streamer and self.streamer.running:
                    frame = self.streamer.get_frame()
                    
                    if frame:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(frame)}\r\n'.encode())
                        self.wfile.write(b'\r\n')
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    else:
                        time.sleep(0.01)
                        
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                logger.error(f"Erro no handler: {e}")
        
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            if self.streamer:
                response = {
                    'running': self.streamer.running,
                    'width': self.streamer.width,
                    'height': self.streamer.height,
                    'frames_received': self.streamer.frame_count
                }
            else:
                response = {'running': False}
            
            import json
            self.wfile.write(json.dumps(response).encode())
        
        else:
            self.send_response(404)
            self.end_headers()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP Server com threading."""
    daemon_threads = True


def run_proxy(ip: str, port: int, username: str, password: str, 
              stream_path: str, local_port: int, preview: bool):
    """Executa o proxy."""
    
    config = StreamConfig(
        ip=ip,
        port=port,
        username=username,
        password=password,
        stream_path=stream_path,
        local_port=local_port
    )
    
    streamer = MJPEGStreamer(config)
    MJPEGHandler.streamer = streamer
    streamer.start()
    
    if preview:
        print("\nAbrindo preview...")
        window_name = "Yoosee Preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        frame_skip = 0
        while True:
            frame_data = streamer.get_frame()
            
            if frame_data:
                if frame_skip % 3 == 0:
                    nparr = np.frombuffer(frame_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    info = f"{streamer.width}x{streamer.height} | Frames: {streamer.frame_count}"
                    cv2.putText(img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    
                    cv2.imshow(window_name, img)
                
                frame_skip += 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)
        
        cv2.destroyAllWindows()
    
    else:
        print(f"\nProxy MJPEG ativo em: http://localhost:{local_port}/stream")
        print("Pressione Ctrl+C para sair\n")
        
        try:
            server = ThreadedHTTPServer(('0.0.0.0', local_port), MJPEGHandler)
            logger.info(f"Servidor iniciado em http://0.0.0.0:{local_port}")
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nParando servidor...")
    
    streamer.stop()


def main():
    parser = argparse.ArgumentParser(description="Proxy RTSP -> MJPEG HTTP")
    parser.add_argument("--ip", type=str, default="192.168.100.49",
                       help="IP da câmera")
    parser.add_argument("--port", type=int, default=554,
                       help="Porta RTSP")
    parser.add_argument("--user", type=str, default="admin",
                       help="Usuário")
    parser.add_argument("--password", type=str, default="HonkaiImpact3rd",
                       help="Senha")
    parser.add_argument("--stream", type=str, default="/onvif1",
                       help="Caminho do stream")
    parser.add_argument("--local-port", type=int, default=8554,
                       help="Porta local do proxy")
    parser.add_argument("--preview", action="store_true",
                       help="Abrir janela de preview")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PROXY RTSP -> MJPEG")
    print("=" * 60)
    print(f"Câmera: {args.ip}:{args.port}{args.stream}")
    print(f"Proxy:  http://localhost:{args.local_port}/stream")
    print("=" * 60)
    
    run_proxy(
        ip=args.ip,
        port=args.port,
        username=args.user,
        password=args.password,
        stream_path=args.stream,
        local_port=args.local_port,
        preview=args.preview
    )


if __name__ == "__main__":
    main()
