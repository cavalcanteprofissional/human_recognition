#!/usr/bin/env python3
"""
Script de teste para conexão com câmera Yoosee via RTSP.
Testa múltiplos endpoints e detecta automaticamente o stream que funciona.
"""

import sys
import socket
import subprocess
from pathlib import Path
from typing import Tuple, Optional, Dict

import cv2
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import YOOSEE_CONFIG, YOOSEE_RTSP_PATHS


def test_rtsp_stream(rtsp_url: str, timeout: int = 5) -> Dict:
    """
    Testa uma URL RTSP específica.
    
    Args:
        rtsp_url: URL RTSP completa
        timeout: Tempo máximo de espera por frame
        
    Returns:
        Dicionário com resultado do teste
    """
    print(f"\nTestando: {rtsp_url}")
    
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        cap.release()
        return {"success": False, "error": "VideoCapture não abriu", "url": rtsp_url}
    
    start_time = time.time()
    frame_count = 0
    fps = 0
    width, height = 0, 0
    
    while time.time() - start_time < timeout:
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_count += 1
            if frame_count == 1:
                width = frame.shape[1]
                height = frame.shape[0]
            if frame_count >= 5:
                fps = frame_count / (time.time() - start_time)
                break
    
    cap.release()
    
    if frame_count > 0:
        return {
            "success": True,
            "url": rtsp_url,
            "width": width,
            "height": height,
            "fps": fps,
            "frames_received": frame_count
        }
    else:
        return {"success": False, "error": "Conectou mas não recebeu frames", "url": rtsp_url}


def test_port_connectivity(ip: str, port: int = 554, timeout: float = 2.0) -> Tuple[bool, str]:
    """Testa conectividade básica a uma porta."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        
        if result == 0:
            return True, "Porta aberta"
        else:
            return False, f"Conexão recusada (código {result})"
    except socket.timeout:
        return False, "Timeout"
    except socket.gaierror:
        return False, "Host não encontrado"
    except Exception as e:
        return False, f"Erro: {str(e)}"


def diagnose_connection(ip: str, port: int = 554, username: str = "admin", 
                        password: str = "123") -> Dict:
    """
    Diagnostica problemas de conexão detalhados.
    
    Returns:
        Dicionário com diagnóstico completo
    """
    print("=" * 60)
    print("DIAGNÓSTICO DE CONEXÃO - CAMERA YOOSEE")
    print("=" * 60)
    print(f"IP: {ip}")
    print(f"Porta: {port}")
    print(f"Usuário: {username}")
    print(f"Senha: {'*' * len(password) if password else '(vazia)'}")
    print("=" * 60)
    
    results = {
        "ip": ip,
        "port": port,
        "steps": []
    }
    
    print("\n[1/4] Testando ping...")
    try:
        result = subprocess.run(["ping", "-n", "1", "-w", "1000", ip],
                              capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            print("    [OK] Host responde a ping")
            results["steps"].append(("ping", True, "Host responde"))
        else:
            print("    [X] Host não responde a ping")
            results["steps"].append(("ping", False, "Sem resposta"))
    except Exception as e:
        print(f"    [X] Erro no ping: {e}")
        results["steps"].append(("ping", False, str(e)))
    
    print("\n[2/4] Testando conectividade na porta RTSP (554)...")
    success, msg = test_port_connectivity(ip, 554)
    print(f"    {'[OK]' if success else '[X]'} {msg}")
    results["steps"].append(("rtsp_port", success, msg))
    
    if success:
        print("\n[3/4] Testando protocolos RTSP...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((ip, 554))
            
            request = "OPTIONS rtsp://{} RTSP/1.0\r\nCSeq: 1\r\n\r\n".format(ip)
            sock.send(request.encode())
            response = sock.recv(1024).decode()
            sock.close()
            
            if 'RTSP' in response:
                print("    [OK] Servidor RTSP responde")
                results["steps"].append(("rtsp_protocol", True, "Servidor RTSP OK"))
            else:
                print("    [X] Servidor não reconhece RTSP")
                results["steps"].append(("rtsp_protocol", False, "Não é RTSP"))
        except Exception as e:
            print(f"    [X] Erro no protocolo RTSP: {e}")
            results["steps"].append(("rtsp_protocol", False, str(e)))
    
    print("\n[4/4] Testando streams RTSP...")
    
    working_streams = []
    for stream_name, path in YOOSEE_RTSP_PATHS.items():
        if username and password:
            url = f"rtsp://{username}:{password}@{ip}:{port}{path}"
        else:
            url = f"rtsp://{ip}:{port}{path}"
        
        result = test_rtsp_stream(url, timeout=3)
        result["stream_name"] = stream_name
        
        if result["success"]:
            print(f"    [OK] {stream_name}: FUNCIONOU ({result['width']}x{result['height']})")
            working_streams.append(result)
        else:
            print(f"    [X] {stream_name}: {result.get('error', 'falhou')}")
    
    results["working_streams"] = working_streams
    
    print("\n" + "=" * 60)
    print("RESUMO DO DIAGNÓSTICO")
    print("=" * 60)
    
    all_steps_ok = all(step[1] for step in results["steps"])
    
    if all_steps_ok and working_streams:
        print("\n[OK] Conexão funcionando!")
        best = working_streams[0]
        print(f"\nStream recomendado: {best['stream_name']}")
        print(f"URL: {best['url']}")
        results["status"] = "success"
        results["best_stream"] = best
    else:
        print("\n[X] Problemas detectados:")
        for step_name, success, msg in results["steps"]:
            if not success:
                print(f"  - {step_name}: {msg}")
        
        if not any(step[0] == "rtsp_port" and step[1] for step in results["steps"]):
            print("\nSugestões:")
            print("  1. Verifique se a câmera está na mesma rede")
            print("  2. Cheque o firewall (porta 554)")
            print("  3. Use o app Yoosee para verificar o IP correto")
        elif not working_streams:
            print("\nSugestões:")
            print("  1. Verifique usuário/senha corretos")
            print("  2. A câmera pode estar em outra sub-rede")
        
        results["status"] = "failed"
    
    return results


def test_all_streams(ip: str, username: str, password: str, port: int = 554):
    """Testa todos os endpoints RTSP disponíveis."""
    print("=" * 60)
    print("TESTE DE CONEXAO RTSP - CAMERA YOOSEE")
    print("=" * 60)
    print(f"IP: {ip}:{port}")
    print(f"Usuario: {username}")
    print(f"Senha: {'*' * len(password) if password else '(vazia)'}")
    print("=" * 60)
    
    results = []
    
    for stream_name, path in YOOSEE_RTSP_PATHS.items():
        if username and password:
            url = f"rtsp://{username}:{password}@{ip}:{port}{path}"
        else:
            url = f"rtsp://{ip}:{port}{path}"
        
        result = test_rtsp_stream(url)
        result["stream_name"] = stream_name
        result["path"] = path
        results.append(result)
        
        if result["success"]:
            print(f"[OK] {stream_name:12} -> FUNCIONOU! ({result['width']}x{result['height']} @ {result['fps']:.1f}fps)")
        else:
            print(f"[X] {stream_name:12} -> FALHOU ({result.get('error', 'desconhecido')})")
    
    print("\n" + "=" * 60)
    print("RESUMO DOS TESTES")
    print("=" * 60)
    
    working_streams = [r for r in results if r["success"]]
    
    if working_streams:
        print(f"\n{len(working_streams)} stream(s) funcionou(aram):\n")
        for r in working_streams:
            print(f"   * {r['stream_name']}: {r['url']}")
        
        best = working_streams[0]
        print(f"\nRecomendado: {best['stream_name']}")
        print(f"\nConfigure no .env:")
        print(f"   YOOSEE_IP={ip}")
        print(f"   YOOSEE_PORT={port}")
        print(f"   YOOSEE_USERNAME={username}")
        print(f"   YOOSEE_PASSWORD={password}")
        print(f"   YOOSEE_STREAM={best['stream_name']}")
        
        return best
    else:
        print("\nNenhum stream funcionou!")
        print("\nPossiveis causas:")
        print("   1. IP incorreto - verifique no app Yoosee")
        print("   2. Credenciais erradas - use as credenciais do app")
        print("   3. Firewall bloqueando a porta 554")
        print("   4. Camera nao suporta RTSP (modelo muito antigo)")
        print("   5. Rede - camera em outra sub-rede")
        
        return None


def test_with_preview(ip: str, username: str, password: str, stream: str = "onvif1", port: int = 554):
    """Testa conexão com preview em tempo real."""
    path = YOOSEE_RTSP_PATHS.get(stream, "/onvif1")
    
    if username and password:
        url = f"rtsp://{username}:{password}@{ip}:{port}{path}"
    else:
        url = f"rtsp://{ip}:{port}{path}"
    
    print(f"\nAbrindo preview do stream '{stream}'...")
    print(f"   URL: {url}")
    print("   Pressione 'q' para sair\n")
    
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("❌ Não foi possível conectar")
        return
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Perda de conexão")
            break
        
        frame_count += 1
        elapsed = time.time() - start_time
        
        if frame_count % 30 == 0:
            fps = frame_count / elapsed
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Yoosee Camera Preview", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    print(f"\nEstatisticas:")
    print(f"   Frames recebidos: {frame_count}")
    print(f"   Tempo: {total_time:.1f}s")
    print(f"   FPS medio: {frame_count/total_time:.1f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Teste de conexão RTSP Yoosee")
    parser.add_argument("--ip", type=str, default=None,
                       help="IP da câmera (ou use .env)")
    parser.add_argument("--user", type=str, default=None,
                       help="Usuário (ou use .env)")
    parser.add_argument("--password", type=str, default=None,
                       help="Senha (ou use .env)")
    parser.add_argument("--port", type=int, default=554,
                       help="Porta RTSP (padrão: 554)")
    parser.add_argument("--diagnose", action="store_true",
                       help="Executar diagnóstico completo")
    parser.add_argument("--preview", action="store_true",
                       help="Abrir preview em tempo real")
    parser.add_argument("--stream", type=str, default="onvif1",
                       help="Stream para preview (padrão: onvif1)")
    
    args = parser.parse_args()
    
    ip = args.ip or YOOSEE_CONFIG.get("ip", "")
    username = args.user or YOOSEE_CONFIG.get("username", "admin")
    password = args.password or YOOSEE_CONFIG.get("password", "")
    port = args.port
    
    if not ip or ip in ["192.168.1.100", "192.168.100.46"]:
        print("❌ IP da câmera não configurado!")
        print("\nPara encontrar o IP:")
        print("  1. Abra o app Yoosee")
        print("  2. Vá nas configurações da câmera")
        print("  3. Procure por 'Informação do dispositivo' ou 'Network'")
        print("  4. Anote o endereço IP")
        print("\nOu execute: python tools/find_yoosee_ip.py")
        return
    
    if args.diagnose:
        diagnose_connection(ip, port, username, password)
    elif args.preview:
        test_with_preview(ip, username, password, args.stream, port)
    else:
        test_all_streams(ip, username, password, port)


if __name__ == "__main__":
    main()
