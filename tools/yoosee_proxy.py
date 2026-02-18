#!/usr/bin/env python3
"""
Proxy RTSP to HTTP com suporte a autenticação Digest.
Este script faz:
1. Autenticação Digest com a câmera
2. Estabelece sessão RTSP
3. Faz parse do stream RTP
4. Serve via HTTP como MJPEG
O OpenCV pode então conectar via HTTP.
"""

import socket
import struct
import threading
import hashlib
import time
import logging
import io
from typing import Optional, Tuple
from http.client import HTTPConnection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YooseeRTSPProxy:
    """Proxy RTSP -> HTTP com autenticação Digest."""
    
    def __init__(self, ip: str, port: int, username: str, password: str,
                 stream_path: str = "onvif1", http_port: int = 8554):
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.stream_path = stream_path
        self.http_port = http_port
        
        self.running = False
        self.server_socket: Optional[socket.socket] = None
        self.camera_socket: Optional[socket.socket] = None
        self.realm: Optional[str] = None
        self.nonce: Optional[str] = None
        self.session: Optional[str] = None
        
    def _send_request(self, method: str, uri: str, extra_headers: str = "") -> str:
        """Envia requisição RTSP."""
        cseq = getattr(self, '_cseq', 1)
        setattr(self, '_cseq', cseq + 1)
        
        request = f"{method} {uri} RTSP/1.0\r\nCSeq: {cseq}\r\n"
        
        if self.session:
            request += f"Session: {self.session}\r\n"
        
        if extra_headers:
            request += extra_headers
            
        request += "\r\n"
        
        self.camera_socket.send(request.encode())
        response = self.camera_socket.recv(16384).decode('utf-8', errors='replace')
        
        return response
    
    def _get_digest_params(self) -> bool:
        """Obtém realm e nonce."""
        try:
            self.camera_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.camera_socket.settimeout(5)
            self.camera_socket.connect((self.ip, self.port))
            
            uri = f"rtsp://{self.ip}:{self.port}/{self.stream_path}"
            
            # OPTIONS
            self._send_request("OPTIONS", uri)
            
            # DESCRIBE
            response = self._send_request("DESCRIBE", uri)
            
            import re
            realm_match = re.search(r'realm="([^"]+)"', response)
            nonce_match = re.search(r'nonce="([^"]+)"', response)
            
            if realm_match and nonce_match:
                self.realm = realm_match.group(1)
                self.nonce = nonce_match.group(1)
                logger.info(f"Digest: realm={self.realm}")
                return True
            
        except Exception as e:
            logger.error(f"Erro: {e}")
        
        return False
    
    def _create_auth(self, method: str, uri: str) -> str:
        """Cria autenticação Digest."""
        ha1 = hashlib.md5(f"{self.username}:{self.realm}:{self.password}".encode()).hexdigest()
        ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
        response = hashlib.md5(f"{ha1}:{self.nonce}:{ha2}".encode()).hexdigest()
        
        return (f'Digest username="{self.username}", realm="{self.realm}", '
                f'nonce="{self.nonce}", uri="{uri}", response="{response}"')
    
    def _setup_rtsp(self) -> bool:
        """Estabelece conexão RTSP completa."""
        try:
            uri = f"rtsp://{self.ip}:{self.port}/{self.stream_path}"
            auth = self._create_auth("DESCRIBE", uri)
            
            # DESCRIBE com auth
            response = self._send_request("DESCRIBE", uri, f"Authorization: {auth}\r\n")
            
            if "200" not in response.split('\r\n')[0]:
                logger.error(f"DESCRIBE falhou")
                return False
            
            # SETUP para vídeo
            auth = self._create_auth("SETUP", f"{uri}/trackID=0")
            response = self._send_request(
                "SETUP", f"{uri}/trackID=0",
                f"Authorization: {auth}\r\nTransport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n"
            )
            
            if "200" not in response.split('\r\n')[0]:
                logger.error(f"SETUP falhou: {response[:200]}")
                return False
            
            import re
            session_match = re.search(r'Session: ([^;]+)', response)
            if not session_match:
                logger.error("Session não encontrada")
                return False
            
            self.session = session_match.group(1)
            
            # PLAY
            auth = self._create_auth("PLAY", uri)
            response = self._send_request(
                "PLAY", uri,
                f"Authorization: {auth}\r\nRange: npt=0.000-\r\n"
            )
            
            logger.info("RTSP conectado!")
            return True
            
        except Exception as e:
            logger.error(f"Erro RTSP: {e}")
            return False
    
    def _parse_rtp(self, data: bytes) -> Optional[bytes]:
        """Parseia RTP packet e retorna JPEG se presente."""
        if len(data) < 12:
            return None
        
        # RTP header
        # Byte 0: version(2) padding(1) extension(1) CSRC count(4)
        # Byte 1: marker(1) payload type(7)
        # Bytes 2-3: sequence number
        # Bytes 4-7: timestamp
        # Bytes 8-11: SSRC
        
        payload_type = data[1] & 0x7F
        
        # 26 = JPEG
        if payload_type != 26:
            return None
        
        # Verificar marker bit
        marker = data[1] & 0x80
        
        # Offset para JPEG payload
        # RTP header = 12 bytes + optional CSRC
        offset = 12
        
        # Verificar se há JPEG payload
        if len(data) < offset + 4:
            return None
        
        # Tipo de JPEG (deve ser 1 para Jpeg)
        jpeg_type = data[offset]
        
        if jpeg_type != 1:
            return None
        
        # Tamanho do header JPEG
        q = data[offset + 1]  # Quantization
        width = data[offset + 2] * 8
        height = data[offset + 3] * 8
        
        # JPEG data starts after 8 bytes of RTP/JPEG header
        jpeg_data = data[offset + 8:]
        
        # Adicionar JPEG SOI e marker
        jpeg = b'\xff\xd8' + jpeg_data
        
        return jpeg
    
    def handle_client(self, client_socket: socket.socket):
        """Manipula cliente HTTP."""
        try:
            request = client_socket.recv(4096).decode('utf-8', errors='replace')
            
            if "/mjpeg" in request or "/stream" in request:
                # MJPEG stream
                response = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                    "Cache-Control: no-cache\r\n"
                    "Connection: keep-alive\r\n"
                    "\r\n"
                )
                client_socket.send(response.encode())
                
                # Buffer para reassembly de frames
                frame_buffer = b''
                
                while self.running:
                    try:
                        data = self.camera_socket.recv(65536)
                        if not data:
                            break
                        
                        # RTP over TCP usa interleaving
                        # Byte 0: '$' (0x24)
                        # Byte 1: channel
                        # Bytes 2-3: length
                        
                        i = 0
                        while i < len(data) and self.running:
                            if data[i] == 0x24 and i + 3 < len(data):
                                channel = data[i + 1]
                                length = struct.unpack("!H", data[i+2:i+4])[0]
                                
                                if i + 4 + length <= len(data):
                                    payload = data[i+4:i+4+length]
                                    
                                    # Channel 0 = video
                                    if channel == 0:
                                        jpeg = self._parse_rtp(payload)
                                        if jpeg and len(jpeg) > 100:
                                            # Enviar frame JPEG
                                            frame = (b'--frame\r\n'
                                                    b'Content-Type: image/jpeg\r\n\r\n' 
                                                    + jpeg + b'\r\n')
                                            try:
                                                client_socket.send(frame)
                                            except:
                                                break
                                    
                                    i += 4 + length
                                else:
                                    break
                            else:
                                i += 1
                                
                    except socket.timeout:
                        continue
                    except Exception as e:
                        break
            
            else:
                # Página simples
                response = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/html\r\n"
                    "Connection: close\r\n"
                    "\r\n"
                    "<html><body>"
                    "<h1>Yoosee Camera Proxy</h1>"
                    "<p>Stream: <a href='/mjpeg'>/mjpeg</a></p>"
                    "</body></html>"
                )
                client_socket.send(response.encode())
                
        except Exception as e:
            logger.error(f"Erro cliente: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    def start(self) -> bool:
        """Inicia o servidor."""
        logger.info("Iniciando proxy...")
        
        if not self._get_digest_params():
            logger.error("Falha ao obter params")
            return False
        
        if not self._setup_rtsp():
            logger.error("Falha ao conectar RTSP")
            return False
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.http_port))
        self.server_socket.listen(5)
        
        self.running = True
        logger.info(f"Proxy ativo em http://localhost:{self.http_port}/mjpeg")
        
        while self.running:
            try:
                self.server_socket.settimeout(1)
                try:
                    client, addr = self.server_socket.accept()
                    thread = threading.Thread(target=self.handle_client, args=(client,))
                    thread.daemon = True
                    thread.start()
                except socket.timeout:
                    continue
            except Exception as e:
                if self.running:
                    logger.error(f"Erro: {e}")
                break
        
        return True
    
    def stop(self):
        """Para o servidor."""
        self.running = False
        
        if self.camera_socket:
            try:
                self.camera_socket.close()
            except:
                pass
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        logger.info("Proxy parado")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Yoosee RTSP -> HTTP Proxy")
    parser.add_argument("--ip", type=str, default="192.168.100.47")
    parser.add_argument("--port", type=int, default=554)
    parser.add_argument("--user", type=str, default="admin")
    parser.add_argument("--password", type=str, default="HonkaiImpact3rd")
    parser.add_argument("--stream", type=str, default="onvif1")
    parser.add_argument("--http-port", type=int, default=8554)
    
    args = parser.parse_args()
    
    proxy = YooseeRTSPProxy(
        args.ip, args.port,
        args.user, args.password,
        args.stream, args.http_port
    )
    
    print(f"Iniciando proxy para {args.ip}")
    print(f"MJPEG: http://localhost:{args.http_port}/mjpeg")
    print("Ctrl+C para parar")
    
    try:
        proxy.start()
    except KeyboardInterrupt:
        print("\nParando...")
        proxy.stop()


if __name__ == "__main__":
    main()
