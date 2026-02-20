#!/usr/bin/env python3
"""
Cliente RTSP com suporte a autenticação Basic e Digest.
Suporta streams H.265 e H.264.
"""

import socket
import struct
import hashlib
import base64
import re
import time
import threading
import logging
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
from enum import Enum

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class AuthType(Enum):
    NONE = "none"
    BASIC = "basic"
    DIGEST = "digest"


@dataclass
class RTSPConfig:
    ip: str
    port: int
    username: str
    password: str
    stream_path: str = "/onvif1"
    use_tcp: bool = True
    debug: bool = False


@dataclass
class RTPPacket:
    """Pacote RTP parseado."""
    version: int
    padding: bool
    extension: bool
    csrc_count: int
    marker: int
    payload_type: int
    sequence_number: int
    timestamp: int
    ssrc: int
    payload: bytes


class RTSPClient:
    """
    Cliente RTSP com autenticação Basic/Digest.
    Suporta H.265 e H.264.
    """
    
    RTP_VERSION = 2
    
    def __init__(self, config: RTSPConfig):
        self.config = config
        self.socket: Optional[socket.socket] = None
        self.auth_type = AuthType.NONE
        self.realm: Optional[str] = None
        self.nonce: Optional[str] = None
        self.session: Optional[str] = None
        self.cseq = 1
        self.rtp_port = 0
        self.rtcp_port = 0
        self.running = False
        self._receive_thread: Optional[threading.Thread] = None
        self._frame_callback: Optional[Callable[[bytes], None]] = None
    
    def _create_socket(self) -> socket.socket:
        """Cria socket TCP para conexão RTSP."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((self.config.ip, self.config.port))
        return sock
    
    def _send_request(self, method: str, uri: str, extra_headers: dict = None) -> str:
        """Envia requisição RTSP."""
        request_lines = [
            f"{method} {uri} RTSP/1.0",
            f"CSeq: {self.cseq}",
            f"User-Agent: PythonRTSPClient/1.0",
        ]
        
        if self.config.debug:
            logger.debug(f"Request: {method} {uri}")
        
        if self.session:
            request_lines.append(f"Session: {self.session}")
        
        if self.auth_type == AuthType.BASIC:
            auth_str = base64.b64encode(
                f"{self.config.username}:{self.config.password}".encode()
            ).decode()
            request_lines.append(f"Authorization: Basic {auth_str}")
        
        elif self.auth_type == AuthType.DIGEST and self.realm and self.nonce:
            auth_header = self._create_digest_auth(method, uri)
            if auth_header:
                request_lines.append(f"Authorization: {auth_header}")
        
        if extra_headers:
            for key, value in extra_headers.items():
                request_lines.append(f"{key}: {value}")
        
        request_lines.append("")
        request_lines.append("")
        
        request = "\r\n".join(request_lines)
        
        if self.config.debug:
            logger.debug(f"→ {request[:200]}...")
        
        self.socket.send(request.encode())
        self.cseq += 1
        
        return self._receive_response()
    
    def _receive_response(self) -> str:
        """Recebe resposta RTSP."""
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = self.socket.recv(4096)
            if not chunk:
                break
            response += chunk
        
        return response.decode('utf-8', errors='replace')
    
    def _parse_response(self, response: str) -> dict:
        """Parseia resposta RTSP."""
        lines = response.split("\r\n")
        
        status_line = lines[0]
        match = re.match(r"RTSP/1\.0 (\d{3}) (.*)", status_line)
        
        if not match:
            return {"status": 0, "message": "Invalid response", "headers": {}}
        
        status = int(match.group(1))
        message = match.group(2)
        
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        
        return {
            "status": status,
            "message": message,
            "headers": headers,
            "body": "\r\n".join(lines[lines.index("") + 1:] if "" in lines else lines[1:])
        }
    
    def _create_digest_auth(self, method: str, uri: str) -> Optional[str]:
        """Cria header de autenticação Digest."""
        if not self.realm or not self.nonce:
            return None
        
        ha1 = hashlib.md5(
            f"{self.config.username}:{self.realm}:{self.config.password}".encode()
        ).hexdigest()
        
        ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
        
        response = hashlib.md5(
            f"{ha1}:{self.nonce}:{ha2}".encode()
        ).hexdigest()
        
        return (
            f'Digest username="{self.config.username}", '
            f'realm="{self.realm}", nonce="{self.nonce}", '
            f'uri="{uri}", response="{response}"'
        )
    
    def _parse_auth_challenge(self, headers: dict) -> bool:
        """Parseia desafio de autenticação e determina tipo."""
        www_auth = headers.get("www-authenticate", "")
        
        if not www_auth:
            return False
        
        if "Digest" in www_auth:
            self.auth_type = AuthType.DIGEST
            realm_match = re.search(r'realm="([^"]+)"', www_auth)
            nonce_match = re.search(r'nonce="([^"]+)"', www_auth)
            
            if realm_match:
                self.realm = realm_match.group(1)
            if nonce_match:
                self.nonce = nonce_match.group(1)
            
            logger.info(f"Digest Auth - Realm: {self.realm}, Nonce: {self.nonce[:20]}...")
            return True
        
        elif "Basic" in www_auth:
            self.auth_type = AuthType.BASIC
            logger.info("Basic Auth suportado")
            return True
        
        return False
    
    def connect(self) -> bool:
        """
        Conecta à câmera RTSP.
        Tenta Basic primeiro, depois Digest se necessário.
        """
        try:
            self.socket = self._create_socket()
            
            response = self._send_request("OPTIONS", f"rtsp://{self.config.ip}{self.config.stream_path}")
            parsed = self._parse_response(response)
            
            if parsed["status"] != 200:
                logger.error(f"OPTIONS falhou: {parsed['status']} {parsed['message']}")
                return False
            
            logger.info("OPTIONS OK")
            
            response = self._send_request(
                "DESCRIBE",
                f"rtsp://{self.config.ip}:{self.config.port}{self.config.stream_path}"
            )
            parsed = self._parse_response(response)
            
            if parsed["status"] == 401:
                logger.info("Autenticação necessária...")
                if self._parse_auth_challenge(parsed["headers"]):
                    response = self._send_request(
                        "DESCRIBE",
                        f"rtsp://{self.config.ip}:{self.config.port}{self.config.stream_path}"
                    )
                    parsed = self._parse_response(response)
            
            if parsed["status"] != 200:
                logger.error(f"DESCRIBE falhou: {parsed['status']} {parsed['message']}")
                return False
            
            logger.info("DESCRIBE OK")
            
            self._extract_media_info(parsed.get("body", ""))
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao conectar: {e}")
            return False
    
    def _extract_media_info(self, sdp: str):
        """Extrai informações de mídia do SDP."""
        if "H265" in sdp or "hevc" in sdp.lower():
            logger.info("Stream: H.265")
        elif "H264" in sdp or "avc" in sdp.lower():
            logger.info("Stream: H.264")
        
        track_match = re.search(r"a=control:trackID=(\d+)", sdp)
        if track_match:
            logger.info(f"Track: {track_match.group(1)}")
    
    def setup(self, track_url: str = None) -> bool:
        """
        Configura o stream RTSP.
        
        Args:
            track_url: URL do track específico (se None, usa stream_path)
        """
        try:
            if self.config.use_tcp:
                self.rtp_port = 0
                self.rtcp_port = 0
                transport = f"Transport: RTP/AVP/TCP;unicast;interleaved=0-1"
            else:
                self.rtp_port = 50000 + (id(self) % 10000)
                self.rtcp_port = self.rtp_port + 1
                transport = f"Transport: RTP/AVP;unicast;client_port={self.rtp_port}-{self.rtcp_port}"
            
            setup_url = track_url or self.config.stream_path
            
            if self.config.debug:
                logger.debug(f"SETUP URL: {setup_url}")
            
            extra_headers = {"Transport": transport}
            
            response = self._send_request(
                "SETUP",
                setup_url,
                {"Transport": transport}
            )
            parsed = self._parse_response(response)
            
            if parsed["status"] == 401:
                logger.info("SETUP precisa de autenticação...")
                if self._parse_auth_challenge(parsed["headers"]):
                    response = self._send_request(
                        "SETUP",
                        setup_url,
                        {"Transport": transport}
                    )
                    parsed = self._parse_response(response)
            
            if parsed["status"] != 200:
                logger.error(f"SETUP falhou: {parsed['status']} {parsed['message']}")
                logger.error(f"Response: {response[:500]}")
                return False
            
            logger.info("SETUP OK")
            return True
            
        except Exception as e:
            logger.error(f"Erro no SETUP: {e}")
            return False
    
    def play(self) -> bool:
        """Inicia reprodução do stream."""
        try:
            response = self._send_request(
                "PLAY",
                f"rtsp://{self.config.ip}:{self.config.port}{self.config.stream_path}"
            )
            parsed = self._parse_response(response)
            
            if parsed["status"] != 200:
                logger.error(f"PLAY falhou: {parsed['status']} {parsed['message']}")
                return False
            
            logger.info("PLAY OK - Stream iniciado")
            return True
            
        except Exception as e:
            logger.error(f"Erro no PLAY: {e}")
            return False
    
    def start_receiving(self, callback: Callable[[bytes], None]):
        """Inicia recebimento de dados RTP em thread separada."""
        self._frame_callback = callback
        self.running = True
        self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receive_thread.start()
        logger.info("Thread de recebimento iniciada")
    
    def _receive_loop(self):
        """Loop de recebimento de dados RTP."""
        buffer = b""
        
        while self.running:
            try:
                if self.config.use_tcp:
                    data = self.socket.recv(65536)
                else:
                    data = self.socket.recv(65536)
                
                if not data:
                    logger.warning("Conexão perdida")
                    break
                
                buffer += data
                
                while len(buffer) >= 4:
                    if buffer[0] == 0x24 and buffer[1] in [0x00, 0x01]:
                        channel = buffer[1]
                        length = struct.unpack(">H", buffer[2:4])[0]
                        
                        if len(buffer) < 4 + length:
                            break
                        
                        rtp_data = buffer[4:4+length]
                        buffer = buffer[4+length:]
                        
                        if channel == 0:
                            self._process_rtp_packet(rtp_data)
                    else:
                        buffer = buffer[1:]
                        
            except Exception as e:
                if self.running:
                    logger.error(f"Erro no recebimento: {e}")
                break
        
        logger.info("Thread de recebimento finalizada")
    
    def _process_rtp_packet(self, data: bytes):
        """Processa pacote RTP e extrai payload de vídeo."""
        if len(data) < 12:
            return
        
        rtp_version = (data[0] >> 6) & 0x03
        rtp_padding = (data[0] >> 5) & 0x01
        rtp_extension = (data[0] >> 4) & 0x01
        rtp_csrc_count = data[0] & 0x0F
        
        rtp_marker = data[1] >> 7
        rtp_payload_type = data[1] & 0x7F
        
        rtp_seq = struct.unpack(">H", data[2:4])[0]
        rtp_timestamp = struct.unpack(">I", data[4:8])[0]
        rtp_ssrc = struct.unpack(">I", data[8:12])[0]
        
        payload_start = 12 + rtp_csrc_count * 4
        
        if rtp_padding:
            padding_length = data[-1]
            payload_end = len(data) - padding_length
        else:
            payload_end = len(data)
        
        video_payload = data[payload_start:payload_end]
        
        if video_payload and self._frame_callback:
            self._frame_callback(video_payload)
    
    def stop(self):
        """Para o stream e fecha conexão."""
        self.running = False
        
        try:
            if self.socket:
                self._send_request("TEARDOWN", 
                    f"rtsp://{self.config.ip}:{self.config.port}{self.config.stream_path}")
        except:
            pass
        
        if self.socket:
            self.socket.close()
            self.socket = None
        
        logger.info("Conexão fechada")


def test_connection(ip: str, port: int, username: str, password: str, stream: str = "/onvif1"):
    """Testa conexão RTSP com a câmera."""
    print("=" * 60)
    print("TESTE DE CONEXÃO RTSP")
    print("=" * 60)
    print(f"IP: {ip}:{port}")
    print(f"Stream: {stream}")
    print(f"Usuário: {username}")
    print("=" * 60)
    
    config = RTSPConfig(
        ip=ip,
        port=port,
        username=username,
        password=password,
        stream_path=stream,
        debug=True
    )
    
    client = RTSPClient(config)
    
    if client.connect():
        print("\n✅ CONECTADO!")
        print(f"Tipo de autenticação: {client.auth_type.value}")
        
        if client.setup(f"rtsp://{ip}:{port}{stream}"):
            print("[OK] SETUP OK")
            
            if client.play():
                print("✅ PLAY OK")
                print("\nRecebendo frames por 5 segundos...")
                
                frame_count = [0]
                start_time = time.time()
                
                def on_frame(data):
                    frame_count[0] += 1
                    if frame_count[0] == 1:
                        print(f"Primeiro frame: {len(data)} bytes")
                
                client.start_receiving(on_frame)
                time.sleep(5)
                client.stop()
                
                elapsed = time.time() - start_time
                print(f"\nFrames recebidos: {frame_count[0]}")
                print(f"Tempo: {elapsed:.1f}s")
                print(f"FPS: {frame_count[0]/elapsed:.1f}")
                
                return True
    
    print("\n❌ FALHA NA CONEXÃO")
    return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("Uso: python rtsp_client.py <ip> <port> <username> <password> [stream]")
        print("Exemplo: python rtsp_client.py 192.168.100.49 554 admin HonkaiImpact3rd /onvif1")
        sys.exit(1)
    
    ip = sys.argv[1]
    port = int(sys.argv[2])
    username = sys.argv[3]
    password = sys.argv[4]
    stream = sys.argv[5] if len(sys.argv) > 5 else "/onvif1"
    
    test_connection(ip, port, username, password, stream)
