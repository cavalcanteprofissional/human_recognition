# Estado do Projeto - Agent Instructions

## Visão Geral do Projeto
Projeto de Visão Computacional para reconhecimento de silhueta humana em tempo real usando LBP + Random Forest.

## Estado Atual: FUNCIONAL
- Treinamento: ✅ Funcionando
- Detecção com Webcam: ✅ Funcionando
- Detecção com Câmera Yoosee: ⚠️ Parcial (ver problemas abaixo)
- Dashboard: ✅ Funcionando

## Configuração Atual (.env)
```
YOOSEE_IP=192.168.100.47
YOOSEE_PORT=554
YOOSEE_USERNAME=admin
YOOSEE_PASSWORD=HonkaiImpact3rd
YOOSEE_STREAM=onvif1
```

## Problema: Câmera Yoosee LB-CA128

### Sintomas
- Porta 554 está ABERTA
- Protocolo RTSP responde (OPTIONS, DESCRIBE)
- Autenticação Digest é necessária (não suportada por OpenCV/FFmpeg)
- Stream H.265 (não JPEG)

### Testes Realizados
```bash
# Teste de conexão básica - PORTA ABERTA
python -c "import socket; s=socket.socket(); s.connect(('192.168.100.47', 554)); print('OK')"

# Teste RTSP - RESPONDE
python tools/test_digest_auth.py
# Resultado: URL encontrada rtsp://admin:HonkaiImpact3rd@192.168.100.47:554/onvif1

# Teste OpenCV - FALHA
python -c "import cv2; c=cv2.VideoCapture('rtsp://admin:HonkaiImpact3rd@192.168.100.47:554/onvif1'); print(c.isOpened())"
# Resultado: False - Nonmatching transport error

# Teste FFmpeg - FALHA
ffprobe -rtsp_transport tcp -i "rtsp://admin:HonkaiImpact3rd@192.168.100.47:554/onvif1"
# Resultado: Nonmatching transport in server reply
```

### Causa Raiz
A câmera usa autenticação Digest que OpenCV/FFmpeg não suportam corretamente. O servidor RTSP também retorna "Nonmatching transport" mesmo com TCP forçado.

### Soluções Possíveis (não implementadas)
1. Proxy RTSP completo com parse de RTP (complexo)
2. Usar app Yoosee para PC (P2P, não RTSP)
3. Configurar câmera para modo HTTP/JPEG (verificar no app)

## Comandos para Teste

### Webcam (funciona)
```bash
python run.py --detect --source webcam
python run.py --detect --source webcam --filter cartoon
```

### Yoosee (não funciona - precisa correção)
```bash
# Auto-discovery
python run.py --auto-find-yoosee

# Teste de conexão
python tools/test_yoosee_connection.py --ip 192.168.100.47 --diagnose

# Detecção (vai falhar)
python run.py --detect --source yoosee
```

### Treinamento
```bash
python run.py --setup    # Baixa dataset
python run.py --train     # Treina modelo
python run.py --dashboard # Abre dashboard
```

## O que Precisa Ser Corrigido

### Prioridade ALTA: Câmera Yoosee
1. Resolver problema de autenticação Digest
2. Implementar proxy RTSP→HTTP completo (parse RTP + autenticação)
3. OU verificar configurações da câmera no app

### Prioridade MÉDIA: Limpeza de código
1. Corrigir warnings de tipo nos arquivos
2. Adicionar testes unitários
3. Documentar APIs

## Arquivos Principais

| Arquivo | Descrição |
|---------|-----------|
| `run.py` | CLI principal |
| `src/config.py` | Configurações e auto-discovery |
| `src/real_time_detector.py` | Detector em tempo real |
| `src/yoosee_camera.py` | Wrapper para câmera Yoosee |
| `tools/find_yoosee_ip.py` | Scanner de rede |
| `tools/test_yoosee_connection.py` | Teste de conexão |
| `tools/rtsp_gateway.py` | Gateway FFmpeg (não funciona com Digest) |
| `tools/yoosee_proxy.py` | Proxy customizado (incompleto) |

## Notas para o Agent
- O projeto já foi treinado (modelos em models/)
- Webcam funciona perfeitamente
- Yoosee precisa de implementação de proxy com autenticação Digest
- FFmpeg está instalado em: `C:\Users\muito\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\`
