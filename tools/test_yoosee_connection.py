#!/usr/bin/env python3
"""
Script de teste para conexão com câmera Yoosee via RTSP.
Testa múltiplos endpoints e detecta automaticamente o stream que funciona.
"""

import cv2
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import YOOSEE_CONFIG, YOOSEE_RTSP_PATHS


def test_rtsp_stream(rtsp_url: str, timeout: int = 5) -> dict:
    """
    Testa uma URL RTSP específica.
    
    Args:
        rtsp_url: URL RTSP completa
        timeout: Tempo máximo de espera por frame
        
    Returns:
        Dicionário com resultado do teste
    """
    print(f"\nTestando: {rtsp_url}")
    
    cap = cv2.VideoCapture(rtsp_url)
    
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


def test_all_streams(ip: str, username: str, password: str, port: int = 554):
    """
    Testa todos os endpoints RTSP disponíveis.
    
    Args:
        ip: Endereço IP da câmera
        username: Usuário
        password: Senha
        port: Porta RTSP
    """
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
    """
    Testa conexão com preview em tempo real.
    
    Args:
        ip: IP da câmera
        username: Usuário
        password: Senha
        stream: Stream a testar
        port: Porta RTSP
    """
    path = YOOSEE_RTSP_PATHS.get(stream, "/onvif1")
    
    if username and password:
        url = f"rtsp://{username}:{password}@{ip}:{port}{path}"
    else:
        url = f"rtsp://{ip}:{port}{path}"
    
    print(f"\nAbrindo preview do stream '{stream}'...")
    print(f"   URL: {url}")
    print("   Pressione 'q' para sair\n")
    
    cap = cv2.VideoCapture(url)
    
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
    parser.add_argument("--preview", action="store_true",
                       help="Abrir preview em tempo real")
    parser.add_argument("--stream", type=str, default="onvif1",
                       help="Stream para preview (padrão: onvif1)")
    
    args = parser.parse_args()
    
    # Usar valores do .env se não fornecidos
    ip = args.ip or YOOSEE_CONFIG.get("ip", "")
    username = args.user or YOOSEE_CONFIG.get("username", "admin")
    password = args.password or YOOSEE_CONFIG.get("password", "")
    port = args.port
    
    if not ip or ip == "192.168.1.100":
        print("❌ IP da câmera não configurado!")
        print("\nPara encontrar o IP:")
        print("  1. Abra o app Yoosee")
        print("  2. Vá nas configurações da câmera")
        print("  3. Procure por 'Informação do dispositivo' ou 'Network'")
        print("  4. Anote o endereço IP")
        print("\nOu execute: python tools/find_yoosee_ip.py")
        return
    
    if args.preview:
        test_with_preview(ip, username, password, args.stream, port)
    else:
        test_all_streams(ip, username, password, port)


if __name__ == "__main__":
    main()
