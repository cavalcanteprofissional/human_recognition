#!/usr/bin/env python3
"""
Script para testar conexão com câmera Yoosee usando autenticação Digest manual.
"""

import socket
import hashlib
import time

IP = "192.168.100.47"
PORT = 554
USER = "admin"
PASSWORD = "HonkaiImpact3rd"


def create_digest_auth(method: str, uri: str, realm: str, nonce: str) -> str:
    """Cria cabeçalho de autenticação Digest."""
    ha1 = hashlib.md5(f"{USER}:{realm}:{PASSWORD}".encode()).hexdigest()
    ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
    response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
    
    return f'Digest username="{USER}", realm="{realm}", nonce="{nonce}", uri="{uri}", response="{response}"'


def send_rtsp_request(sock, method: str, uri: str, auth: str = None) -> str:
    """Envia requisição RTSP e retorna resposta."""
    request = f"{method} {uri} RTSP/1.0\r\nCSeq: 1\r\n"
    
    if auth:
        request += f"Authorization: {auth}\r\n"
    
    request += "\r\n"
    
    sock.send(request.encode())
    response = sock.recv(4096).decode('utf-8', errors='replace')
    
    return response


def get_stream_url():
    """Tenta obter URL do stream via RTSP com autenticação Digest."""
    
    paths = ["/onvif1", "/onvif2", "/live.sdp", "/11"]
    
    for path in paths:
        print(f"\n=== Testando {path} ===")
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((IP, PORT))
            
            # OPTIONS
            response = send_rtsp_request(sock, f"OPTIONS rtsp://{IP}", f"rtsp://{IP}:554{path}")
            print(f"OPTIONS: {response.split()[1]}")
            
            # DESCRIBE sem auth
            response = send_rtsp_request(sock, f"DESCRIBE rtsp://{IP}:554{path}", f"rtsp://{IP}:554{path}")
            print(f"DESCRIBE (sem auth): {response.split()[1]}")
            
            # Se precisa de autenticação
            if "401" in response:
                # Extrair realm e nonce
                import re
                realm_match = re.search(r'realm="([^"]+)"', response)
                nonce_match = re.search(r'nonce="([^"]+)"', response)
                
                if realm_match and nonce_match:
                    realm = realm_match.group(1)
                    nonce = nonce_match.group(1)
                    
                    print(f"Realm: {realm}")
                    print(f"Nonce: {nonce}")
                    
                    # Criar autenticação
                    uri = f"rtsp://{IP}:554{path}"
                    auth = create_digest_auth("DESCRIBE", uri, realm, nonce)
                    
                    # DESCRIBE com auth
                    response = send_rtsp_request(sock, f"DESCRIBE rtsp://{IP}:554{path}", uri, auth)
                    print(f"DESCRIBE (com auth): {response.split()[1]}")
                    
                    if "200" in response:
                        # Tentar SETUP e PLAY
                        # Encontrar sessão na resposta
                        import re
                        session_match = re.search(r'Session: ([^;]+)', response)
                        
                        if session_match or "sdp" in response.lower():
                            # Extrair URL do stream da resposta SDP
                            media_match = re.search(r'media url=([^\s]+)', response, re.IGNORECASE)
                            control_match = re.search(r'control:([^\r\n]+)', response)
                            
                            print(f"\n*** STREAM ENCONTRADO! ***")
                            print(f"Caminho: {path}")
                            print(f"URL RTSP: rtsp://{USER}:{PASSWORD}@{IP}:554{path}")
                            
                            sock.close()
                            return f"rtsp://{USER}:{PASSWORD}@{IP}:554{path}"
            
            sock.close()
            
        except Exception as e:
            print(f"Erro: {e}")
    
    return None


if __name__ == "__main__":
    print("=== Teste de conexão RTSP com autenticação Digest ===")
    print(f"IP: {IP}")
    print(f"Porta: {PORT}")
    print(f"Usuário: {USER}")
    print(f"Senha: {PASSWORD}")
    print()
    
    url = get_stream_url()
    
    if url:
        print(f"\n\nURL DO STREAM: {url}")
    else:
        print("\n\nNão foi possível encontrar o stream.")
