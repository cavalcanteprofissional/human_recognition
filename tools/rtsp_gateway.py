#!/usr/bin/env python3
"""
Gateway RTSP to HTTP usando FFmpeg com suporte a autenticação Digest.
Este script cria um servidor HTTP que faz o bridge RTSP -> HTTP.
"""

import subprocess
import threading
import time
import sys
import socket
import hashlib
import signal
from pathlib import Path
from typing import Optional, Tuple

_process: Optional[subprocess.Popen] = None
_http_port = 8554


def check_ffmpeg() -> Tuple[bool, str]:
    """Check if FFmpeg is available."""
    paths = [
        r"C:\Users\muito\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe",
        "ffmpeg"
    ]
    
    for path in paths:
        try:
            result = subprocess.run([path, "-version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                return True, version_line
        except:
            continue
    
    return False, "FFmpeg not found"


def get_digest_auth(ip: str, port: int, path: str, username: str, password: str) -> str:
    """Faz autenticação Digest e retorna a URL RTSP com as headers corretas."""
    
    # Conectar e obter nonce
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)
    sock.connect((ip, port))
    
    uri = f"rtsp://{ip}:{port}{path}"
    
    # OPTIONS
    request = f"OPTIONS {uri} RTSP/1.0\r\nCSeq: 1\r\n\r\n"
    sock.send(request.encode())
    sock.recv(4096)
    
    # DESCRIBE para obter nonce
    request = f"DESCRIBE {uri} RTSP/1.0\r\nCSeq: 2\r\n\r\n"
    sock.send(request.encode())
    response = sock.recv(4096).decode('utf-8', errors='replace')
    
    import re
    realm_match = re.search(r'realm="([^"]+)"', response)
    nonce_match = re.search(r'nonce="([^"]+)"', response)
    
    sock.close()
    
    if not realm_match or not nonce_match:
        print("Não foi possível obter nonce")
        return uri
    
    realm = realm_match.group(1)
    nonce = nonce_match.group(1)
    
    # Criar autenticação
    ha1 = hashlib.md5(f"{username}:{realm}:{password}".encode()).hexdigest()
    ha2 = hashlib.md5(f"DESCRIBE:{uri}".encode()).hexdigest()
    response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
    
    auth_header = (f'Digest username="{username}", realm="{realm}", nonce="{nonce}", '
                   f'uri="{uri}", response="{response}"')
    
    return uri, auth_header


def start_rtsp_gateway(rtsp_url: str, http_port: int = 8554, 
                       resolution: str = "640x480") -> Tuple[bool, str]:
    """
    Inicia FFmpeg para converter RTSP para HTTP MJPEG.
    """
    global _process, _http_port
    
    http_url = f"http://localhost:{http_port}/stream.mjpg"
    _http_port = http_port
    
    stop_gateway()
    
    print(f"Iniciando gateway RTSP -> HTTP...")
    print(f"  RTSP: {rtsp_url}")
    print(f"  HTTP: {http_url}")
    
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-r", "15",
        "-s", resolution,
        "-codec:v", "mjpeg",
        "-q:v", "5",
        "-f", "mjpeg",
        "-listen", "1",
        f"http://localhost:{http_port}/stream.mjpg"
    ]
    
    # Tentar caminhos alternativos
    ffmpeg_paths = [
        r"C:\Users\muito\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe",
        "ffmpeg"
    ]
    
    for ffmpeg_cmd in ffmpeg_paths:
        try:
            cmd[0] = ffmpeg_cmd
            
            _process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL
            )
            
            time.sleep(3)
            
            if _process.poll() is not None:
                stderr = _process.stderr.read().decode() if _process.stderr else ""
                print(f"FFmpeg saiu: {stderr[:200]}")
                continue
            
            print(f"Gateway iniciado!")
            return True, http_url
            
        except FileNotFoundError:
            print(f"FFmpeg não encontrado: {ffmpeg_cmd}")
            continue
        except Exception as e:
            print(f"Erro: {e}")
            continue
    
    return False, "FFmpeg não disponível"


def stop_gateway():
    """Para o gateway."""
    global _process
    if _process and _process.poll() is None:
        print("Parando gateway...")
        _process.terminate()
        try:
            _process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _process.kill()
        _process = None


def create_digest_rtsp_url(ip: str, port: int, username: str, password: str, 
                          stream: str = "onvif1") -> str:
    """Cria URL RTSP com autenticação Digest via FFmpeg."""
    
    path = f"/{stream}"
    uri, auth_header = get_digest_auth(ip, port, path, username, password)
    
    # FFmpeg usa -rtsp_transport tcp e headers
    # Mas para Digest, precisamos usar um workaround
    # Vamos criar uma URL especial para FFmpeg
    
    # Tentar com usuário e senha direto (funciona em alguns casos)
    url = f"rtsp://{username}:{password}@{ip}:{port}{path}"
    
    return url, auth_header


def start_gateway_with_digest(ip: str, port: int, username: str, password: str,
                             stream: str = "onvif1", http_port: int = 8554) -> Tuple[bool, str]:
    """Inicia gateway com suporte a autenticação Digest."""
    
    # Primeiro tenta URL direta
    url = f"rtsp://{username}:{password}@{ip}:{port}/{stream}"
    print(f"Tentando URL direta: {url}")
    
    success, result = start_rtsp_gateway(url, http_port)
    
    if success:
        return True, result
    
    # Se falhou, tentar com FFmpeg específico para Digest
    print("Tentando com opções extras...")
    
    # Este é um workaround - FFmpeg não suporta Digest nativamente bem
    # Mas vamos tentar de qualquer forma
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-fflags", "+genpts",
        "-stimeout", "10000000",  # 10s timeout
        "-i", url,
        "-r", "15",
        "-s", "640x480",
        "-codec:v", "mjpeg",
        "-q:v", "5",
        "-f", "mjpeg",
        "-listen", "1",
        f"http://localhost:{http_port}/stream.mjpg"
    ]
    
    ffmpeg_paths = [
        r"C:\Users\muito\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe",
        "ffmpeg"
    ]
    
    for ffmpeg_cmd in ffmpeg_paths:
        try:
            cmd[0] = ffmpeg_cmd
            
            global _process
            _process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL
            )
            
            time.sleep(3)
            
            if _process.poll() is not None:
                continue
            
            return True, f"http://localhost:{http_port}/stream.mjpg"
            
        except:
            continue
    
    return False, "Falha ao iniciar gateway"


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Gateway RTSP -> HTTP")
    parser.add_argument("--ip", type=str, default="192.168.100.47")
    parser.add_argument("--port", type=int, default=554)
    parser.add_argument("--user", type=str, default="admin")
    parser.add_argument("--password", type=str, default="HonkaiImpact3rd")
    parser.add_argument("--stream", type=str, default="onvif1")
    parser.add_argument("--http-port", type=int, default=8554)
    
    args = parser.parse_args()
    
    available, version = check_ffmpeg()
    if not available:
        print(f"FFmpeg não encontrado: {version}")
        return
    
    print(f"FFmpeg: {version}")
    
    success, result = start_gateway_with_digest(
        args.ip, args.port, args.user, args.password,
        args.stream, args.http_port
    )
    
    if success:
        print(f"\nGateway ativo em: {result}")
        print("Pressione Ctrl+C para parar")
        
        try:
            signal.signal(signal.SIGINT, lambda s, f: stop_gateway())
            while _process and _process.poll() is None:
                time.sleep(1)
        except KeyboardInterrupt:
            stop_gateway()
            print("\nGateway parado")
    else:
        print(f"Falha: {result}")


if __name__ == "__main__":
    main()
