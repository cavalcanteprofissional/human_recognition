#!/usr/bin/env python3
"""
Utilitário para encontrar o IP da câmera Yoosee na rede local.
Baseado em técnicas de scan de rede.
"""

import socket
import threading
import subprocess
import ipaddress
from concurrent.futures import ThreadPoolExecutor
import time
import requests
from urllib.parse import urlparse

def get_local_network():
    """Obtém o range da rede local."""
    try:
        result = subprocess.run(['ipconfig'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'IPv4' in line or 'IP Address' in line:
                ip = line.split(':')[-1].strip()
                if ip and not ip.startswith('169.254'):
                    network = '.'.join(ip.split('.')[:-1]) + '.0/24'
                    return network
    except:
        pass
    
    return "192.168.1.0/24"


def check_port(ip, port=554, timeout=1):
    """Verifica se a porta está aberta."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False


def check_http(ip, port=80, timeout=2):
    """Verifica se há servidor HTTP (API da câmera)."""
    try:
        url = f"http://{ip}:{port}/"
        response = requests.get(url, timeout=timeout)
        return response.status_code in [200, 401, 403]
    except:
        return False


def check_rtsp(ip, port=554):
    """Tenta fazer uma requisição RTSP básica."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect((ip, port))
        
        request = "OPTIONS rtsp://{} RTSP/1.0\r\nCSeq: 1\r\n\r\n".format(ip)
        sock.send(request.encode())
        
        response = sock.recv(1024).decode()
        sock.close()
        
        if 'RTSP' in response:
            return True
    except Exception as e:
        pass
    return False


def get_device_info_http(ip, port=80):
    """Tenta obter informações do dispositivo via HTTP."""
    try:
        urls_to_try = [
            f"http://{ip}:{port}/",
            f"http://{ip}:{port}/cgi-bin/mjpeg",
            f"http://{ip}:{port}/snapshot.jpg",
        ]
        
        for url in urls_to_try:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    return {"success": True, "url": url, "port": port}
            except:
                continue
        
        return {"success": False, "port": port}
    except Exception as e:
        return {"success": False, "error": str(e)}


def scan_network(network="192.168.1.0/24", ports=None):
    """Escaneia a rede procurando câmeras."""
    if ports is None:
        ports = [554, 80, 8080, 8554, 9000, 8000]
    
    print(f"Escaneando rede {network} nas portas: {ports}...")
    
    network = ipaddress.ip_network(network, strict=False)
    candidates = []
    http_candidates = []
    
    def scan_host(ip):
        ip_str = str(ip)
        
        for port in ports:
            if check_port(ip_str, port, timeout=0.5):
                if port == 554:
                    if check_rtsp(ip_str, port):
                        print(f"[OK] Camera RTSP: {ip_str}:{port}")
                        candidates.append((ip_str, port, "RTSP"))
                elif port in [80, 8080, 8000, 9000]:
                    if check_http(ip_str, port):
                        print(f"[OK] Camera HTTP: {ip_str}:{port}")
                        http_candidates.append((ip_str, port, "HTTP"))
                else:
                    print(f"[OK] Porta aberta: {ip_str}:{port}")
                    candidates.append((ip_str, port, "UNKNOWN"))
                break
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        list(executor.map(scan_host, network.hosts()))
    
    return candidates + http_candidates

def test_rtsp_url(ip, port=554, username="admin", password="123"):
    """Testa URLs RTSP comuns com credenciais."""
    paths = ["/onvif1", "/onvif2", "/live.sdp", "/11", "/h264"]
    
    for path in paths:
        if username and password:
            url = f"rtsp://{username}:{password}@{ip}:{port}{path}"
        else:
            url = f"rtsp://{ip}:{port}{path}"
        
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', url],
                timeout=3,
                capture_output=True
            )
            if result.returncode == 0:
                return url, path.replace('/', '')
        except:
            pass
    
    return None, None


def find_yoosee_camera(network=None, ports=None):
    """
    Encontra a câmera Yoosee na rede local.
    
    Args:
        network: Range de rede (ex: '192.168.1.0/24'). Se None, detecta automaticamente.
        ports: Lista de portas para scan
    
    Returns:
        Tuple (ip, port, stream_type) ou (None, None, None) se não encontrar
    """
    if network is None:
        network = get_local_network()
    
    candidates = scan_network(network, ports)
    
    if not candidates:
        return None, None, None
    
    for ip, port, proto in candidates:
        if proto == "RTSP":
            url, stream_type = test_rtsp_url(ip, port)
            if url:
                return ip, port, stream_type
    
    return None, None, None


def update_env_with_camera_ip(env_path=None):
    """
    Encontra a câmera e atualiza o arquivo .env com o IP encontrado.
    
    Args:
        env_path: Caminho para o arquivo .env. Se None, usa o padrão.
    
    Returns:
        Tuple (ip, port, stream_type) se encontrado, (None, None, None) caso contrário
    """
    from pathlib import Path
    import re
    
    if env_path is None:
        env_path = Path(__file__).parent.parent / ".env"
    else:
        env_path = Path(env_path)
    
    print(f"Procurando câmera Yoosee na rede...")
    ip, port, stream_type = find_yoosee_camera()
    
    if ip is None:
        print("Camera não encontrada na rede.")
        return None, None, None
    
    print(f"Camera encontrada: {ip}:{port} (stream: {stream_type})")
    
    # Lê o .env atual
    if env_path.exists():
        env_content = env_path.read_text()
    else:
        env_content = ""
    
    # Atualiza ou adiciona YOOSEE_IP
    ip_pattern = re.compile(r'^YOOSEE_IP=.*$', re.MULTILINE)
    if ip_pattern.search(env_content):
        env_content = ip_pattern.sub(f"YOOSEE_IP={ip}", env_content)
    else:
        env_content += f"\nYOOSEE_IP={ip}"
    
    # Atualiza ou adiciona YOOSEE_PORT
    port_pattern = re.compile(r'^YOOSEE_PORT=.*$', re.MULTILINE)
    if port_pattern.search(env_content):
        env_content = port_pattern.sub(f"YOOSEE_PORT={port}", env_content)
    else:
        env_content += f"\nYOOSEE_PORT={port}"
    
    # Atualiza ou adiciona YOOSEE_STREAM
    if stream_type:
        stream_pattern = re.compile(r'^YOOSEE_STREAM=.*$', re.MULTILINE)
        if stream_pattern.search(env_content):
            env_content = stream_pattern.sub(f"YOOSEE_STREAM={stream_type}", env_content)
        else:
            env_content += f"\nYOOSEE_STREAM={stream_type}"
    
    # Salva o .env
    env_path.write_text(env_content)
    print(f".env atualizado: YOOSEE_IP={ip}, YOOSEE_PORT={port}, YOOSEE_STREAM={stream_type}")
    
    return ip, port, stream_type

def main():
    print("=== Rede de Varredura para Camera Yoosee ===")
    print("=" * 50)
    
    # Encontrar rede local
    network = get_local_network()
    print(f"Rede local detectada: {network}")
    
    # Scan da rede
    candidates = scan_network(network)
    
    if not candidates:
        print("\nNenhuma camera encontrada.")
        print("\nSugestoes:")
        print("1. Verifique se a camera esta ligada e na mesma rede")
        print("2. Conecte via app Yoosee para confirmar o IP")
        print("3. Tente escanear manualmente ranges comuns:")
        print("   - 192.168.0.0/24")
        print("   - 10.0.0.0/24")
        print("   - 172.16.0.0/24")
        return
    
    print(f"\nEncontradas {len(candidates)} possiveis cameras:")
    
    for i, (ip, port, proto) in enumerate(candidates, 1):
        print(f"\n{i}. {ip}:{port} ({proto})")
        
        if proto == "RTSP":
            url, stream_type = test_rtsp_url(ip, port)
            if url:
                print(f"\nURL encontrada: {url}")
                print("\nAdicione ao .env:")
                print(f"YOOSEE_IP={ip}")
                print(f"YOOSEE_PORT={port}")
                print(f"YOOSEE_STREAM={stream_type}")
                break
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()