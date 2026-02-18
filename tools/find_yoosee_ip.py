#!/usr/bin/env python3
"""
Utilitário para encontrar o IP da câmera Yoosee na rede local.
Baseado em técnicas de scan de rede [citation:4].
"""

import socket
import threading
import subprocess
import ipaddress
from concurrent.futures import ThreadPoolExecutor
import time

def get_local_network():
    """Obtém o range da rede local."""
    try:
        # No Windows
        result = subprocess.run(['ipconfig'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'IPv4' in line or 'IP Address' in line:
                ip = line.split(':')[-1].strip()
                if ip and not ip.startswith('169.254'):  # Ignora APIPA
                    # Assumindo máscara /24
                    network = '.'.join(ip.split('.')[:-1]) + '.0/24'
                    return network
    except:
        pass
    
    # Fallback para range comum
    return "192.168.1.0/24"

def check_port(ip, port=554, timeout=1):
    """Verifica se a porta RTSP está aberta."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False

def check_rtsp(ip, port=554):
    """Tenta fazer uma requisição RTSP básica."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect((ip, port))
        
        # Envia OPTIONS request RTSP
        request = "OPTIONS rtsp://{} RTSP/1.0\r\nCSeq: 1\r\n\r\n".format(ip)
        sock.send(request.encode())
        
        response = sock.recv(1024).decode()
        sock.close()
        
        if 'RTSP' in response:
            return True
    except:
        pass
    return False

def scan_network(network="192.168.1.0/24", ports=[554, 80, 8080]):
    """Escaneia a rede procurando câmeras."""
    print(f"Escaneando rede {network}...")
    
    network = ipaddress.ip_network(network, strict=False)
    candidates = []
    
    def scan_host(ip):
        ip_str = str(ip)
        for port in ports:
            if check_port(ip_str, port):
                if port == 554 and check_rtsp(ip_str, port):
                    print(f"[OK] Camera Yoosee encontrada: {ip_str}:{port} (RTSP)")
                    candidates.append((ip_str, port, "RTSP"))
                elif port in [80, 8080]:
                    print(f"[?] Possivel camera: {ip_str}:{port} (HTTP)")
                    candidates.append((ip_str, port, "HTTP"))
                break
    
    # Scan em paralelo
    with ThreadPoolExecutor(max_workers=50) as executor:
        list(executor.map(scan_host, network.hosts()))
    
    return candidates

def test_rtsp_url(ip, port=554):
    """Testa URLs RTSP comuns."""
    paths = ["/onvif1", "/onvif2", "/live.sdp", "/11", "/h264"]
    
    for path in paths:
        url = f"rtsp://{ip}:{port}{path}"
        print(f"Testando: {url}")
        
        # Teste rápido com ffprobe se disponível
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', url],
                timeout=3,
                capture_output=True
            )
            if result.returncode == 0:
                print(f"  ✅ Funciona: {url}")
                return url
        except:
            pass
    
    return None

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
            url = test_rtsp_url(ip, port)
            if url:
                print(f"\nURL encontrada: {url}")
                print("\nAdicione ao .env:")
                print(f"YOOSEE_IP={ip}")
                print(f"YOOSEE_PORT={port}")
                print(f"YOOSEE_STREAM={url.split('/')[-1]}")
                break
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()