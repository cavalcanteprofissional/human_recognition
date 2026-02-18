👤 Human Recognition Project
<p align="center"> <img src="https://raw.githubusercontent.com/opencv/opencv/master/doc/opencv-logo.png" width="300" alt="OpenCV Logo"/> </p><p align="center"> <strong>Projeto de Visão Computacional para reconhecimento de silhueta humana em tempo real</strong> </p><p align="center"> <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version"/> <img src="https://img.shields.io/badge/OpenCV-4.8+-green.svg" alt="OpenCV"/> <img src="https://img.shields.io/badge/Streamlit-1.28+-red.svg" alt="Streamlit"/> <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/> </p>
📋 Sobre o Projeto
Este projeto implementa um sistema completo de reconhecimento de silhueta humana utilizando técnicas clássicas de Visão Computacional e Machine Learning. O sistema é capaz de:

Treinar um classificador Random Forest do zero usando características LBP (Local Binary Patterns)

Detectar presença humana em tempo real via webcam ou câmera IP Yoosee

Aplicar 6 filtros criativos diferentes para visualização estilizada

Visualizar métricas e resultados em um dashboard interativo com Streamlit

🎯 Objetivos Acadêmicos
Este projeto foi desenvolvido como Trabalho Final para a disciplina de Processamento de Imagem e Visão Computacional, atendendo aos seguintes requisitos:

✅ Implementação de algoritmo do zero (não usar soluções prontas)

✅ Dataset público e bem documentado

✅ Extração manual de características (LBP)

✅ Treinamento com variação de hiperparâmetros

✅ Aplicação em tempo real com webcam/câmera IP

✅ Dashboard interativo com métricas e visualizações

✅ Documentação completa do pipeline

🚀 Começando
Pré-requisitos
Python 3.9+

Poetry (gerenciador de dependências)

Webcam ou Câmera IP Yoosee

Conta no Kaggle (para download do dataset)

Instalação
Clone o repositório:

bash
git clone https://github.com/seu-usuario/human_recognition.git
cd human_recognition
Instale as dependências com Poetry:

bash
poetry install
Ative o ambiente virtual:

bash
poetry shell
Configure as variáveis de ambiente:

Crie um arquivo .env na raiz do projeto:

env
# Credenciais Kaggle (obrigatório para download do dataset)
KAGGLE_USERNAME=seu_usuario_kaggle
KAGGLE_KEY=sua_chave_kaggle

# Configurações da Câmera Yoosee (opcional)
YOOSEE_IP=192.168.1.100
YOOSEE_PORT=554
YOOSEE_USERNAME=admin
YOOSEE_PASSWORD=sua_senha
YOOSEE_STREAM=onvif1
📦 Estrutura do Projeto
text
human_recognition/
├── .env                      # Variáveis de ambiente
├── .gitignore                 # Arquivos ignorados pelo git
├── pyproject.toml             # Dependências do Poetry
├── README.md                  # Este arquivo
├── run.py                     # Script principal
├── run_dashboard.py           # Script do dashboard
│
├── data/                      # Dados do projeto
│   ├── raw/                   # Dataset original
│   └── processed/             # Dados processados
│
├── models/                     # Modelos treinados
│   └── .gitkeep
│
├── reports/                    # Relatórios e figuras
│   └── figures/                # Figuras geradas
│
├── src/                        # Código fonte
│   ├── __init__.py
│   ├── config.py               # Configurações
│   ├── data_loader.py          # Carregamento do dataset
│   ├── feature_extractor.py    # Extração LBP
│   ├── train.py                # Treinamento do modelo
│   ├── real_time_detector.py   # Detecção em tempo real
│   ├── yoosee_camera.py        # Integração com câmera Yoosee
│   ├── dashboard.py            # Dashboard Streamlit
│   └── utils.py                # Utilitários
│
└── tools/                       # Ferramentas auxiliares
    └── find_yoosee_ip.py        # Scanner para encontrar câmera Yoosee
🎯 Funcionalidades
1. Pipeline de Machine Learning
Dataset: Human Detection Dataset (Kaggle) com 921 imagens 256x256

Extração de características: LBP (Local Binary Patterns) com 59 features

Classificador: Random Forest com grid search de hiperparâmetros

Métricas: Acurácia, Precisão, Recall, F1-Score, Matriz de Confusão

2. Detecção em Tempo Real
Webcam local: Suporte nativo via OpenCV

Câmera Yoosee: Integração via RTSP/ONVIF com reconexão automática

Baixa latência: Streaming otimizado para tempo real

3. Filtros Criativos
Filtro	Descrição	Exemplo
cartoon	Efeito cartoon com bordas suaves	Desenho animado
edges	Detecção de bordas coloridas (Canny)	Contornos destacados
colormap	Mapas de cor criativos (OCEAN, JET)	Efeito térmico
stylized	Efeito artístico estilizado	Pintura
pencil	Efeito de desenho a lápis	Sketch
none	Sem filtro	Imagem original
4. Dashboard Interativo
📊 Visão Geral: Pipeline completo e explicação do LBP

🤖 Treinamento: Configuração de parâmetros e grid search

🎥 Detecção em Tempo Real: Transmissão ao vivo com estatísticas

📈 Análise de Métricas: Gráficos interativos e matriz de confusão

ℹ️ Sobre: Documentação detalhada do projeto

📊 Dataset
Human Detection Dataset
Fonte: Kaggle - Human Detection Dataset

Características:

Total de imagens: 921

Resolução: 256x256 pixels

Formato: PNG

Classes:

1: Com presença humana

0: Sem presença humana

Divisão dos dados:

Treino: 70% (≈645 imagens)

Validação: 10% (≈92 imagens)

Teste: 20% (≈184 imagens)

🧠 Algoritmos e Técnicas
Local Binary Patterns (LBP)
O LBP é um descritor de textura local que se tornou popular devido à sua eficiência computacional e robustez a variações de iluminação.

Princípio de funcionamento:

Para cada pixel, compara com seus 8 vizinhos em um círculo de raio R

Se vizinho ≥ pixel central → 1, senão → 0

Gera um número binário de 8 bits

Histograma dos padrões forma o vetor de características

Parâmetros utilizados:

Radius: 1

N_points: 8

Method: 'uniform' (reduz para 59 features)

Random Forest
Classificador ensemble que combina múltiplas árvores de decisão.

Hiperparâmetros testados:

n_estimators: [10, 50, 100, 200]

max_depth: [5, 10, 15, None]

min_samples_split: [2, 5, 10]

min_samples_leaf: [1, 2, 4]

📈 Métricas de Avaliação
Métrica	Fórmula	Descrição
Acurácia	(VP + VN) / (VP + VN + FP + FN)	Proporção de acertos totais
Precisão	VP / (VP + FP)	Proporção de positivos corretos
Recall	VP / (VP + FN)	Capacidade de encontrar todos os positivos
F1-Score	2 * (Precisão * Recall) / (Precisão + Recall)	Média harmônica entre precisão e recall
Onde:

VP: Verdadeiros Positivos (humano detectado corretamente)

VN: Verdadeiros Negativos (não humano detectado corretamente)

FP: Falsos Positivos (falso alarme)

FN: Falsos Negativos (humano não detectado)

🎮 Como Usar
1. Setup Inicial (baixar e preparar dados)
bash
poetry run python run.py --setup
2. Treinar Modelo
bash
poetry run python run.py --train
3. Executar Dashboard Interativo (recomendado)
bash
# Via run.py
poetry run python run.py --dashboard

# Ou diretamente
poetry run python run_dashboard.py
Acesse: http://localhost:8501

4. Detecção em Tempo Real (Terminal)
bash
# Com webcam e filtro cartoon (padrão)
poetry run python run.py --detect

# Com filtro específico
poetry run python run.py --detect --filter edges
poetry run python run.py --detect --filter colormap
poetry run python run.py --detect --filter stylized
poetry run python run.py --detect --filter pencil
poetry run python run.py --detect --filter none
5. Análise de Resultados
bash
# Analisar resultados de treinamento específicos
poetry run python run.py --analyze reports/results_20240101_120000.json

# Comparar todos os filtros
poetry run python run.py --compare-filters
📹 Integração com Câmera Yoosee
Encontrar a Câmera na Rede
bash
poetry run python tools/find_yoosee_ip.py
Endpoints RTSP Suportados
rtsp://usuario:senha@ip:554/onvif1 (stream principal)

rtsp://usuario:senha@ip:554/onvif2 (sub-stream)

rtsp://usuario:senha@ip:554/live.sdp

rtsp://usuario:senha@ip:554/11

rtsp://usuario:senha@ip:554/h264

Uso no Dashboard
Selecione "Câmera Yoosee (IP)" na barra lateral

Configure IP, usuário e senha

Clique em "Conectar Yoosee"

Inicie a detecção normalmente

🎨 Galeria de Filtros
O projeto oferece 6 filtros criativos que podem ser aplicados em tempo real:

Cartoon: Efeito de desenho animado com bordas suaves

Edges: Detecção de bordas coloridas (Canny)

Colormap: Mapas de cor (OCEAN, JET, etc.)

Stylized: Efeito artístico estilizado

Pencil: Efeito de desenho a lápis

None: Imagem original sem filtro

📊 Resultados Esperados
Métricas de Referência
Acurácia: > 85%

Precisão: > 80%

Recall: > 80%

F1-Score: > 80%

Performance em Tempo Real
Webcam: ~30 FPS

Yoosee (Wi-Fi): ~15-20 FPS

Latência: < 100ms

🔧 Solução de Problemas
Dataset não baixa
bash
# Verifique as credenciais do Kaggle no .env
# Tente baixar manualmente do site e colocar em data/raw/
Câmera Yoosee não conecta
bash
# 1. Teste o IP com ping
ping 192.168.1.100

# 2. Use o scanner de rede
poetry run python tools/find_yoosee_ip.py

# 3. Teste diferentes streams no dashboard
Dashboard lento
Reduza a resolução da câmera

Feche outras aplicações

Use sub-stream da Yoosee (onvif2)

📝 Relatório Acadêmico
O projeto inclui documentação completa para o relatório:

Introdução: Problema de visão computacional escolhido

Metodologia: Fluxo completo da solução

Dataset: Fonte, divisão e características

Algoritmos: Explicação detalhada do LBP e Random Forest

Experimentos: Variação de hiperparâmetros e resultados

Métricas: Equações e análise de desempenho

Implementação: Detalhes técnicos e código

Resultados: Demonstração em tempo real

Conclusão: Análise crítica e trabalhos futuros

🤝 Contribuições
Contribuições são bem-vindas! Siga os passos:

Fork o projeto

Crie sua feature branch (git checkout -b feature/AmazingFeature)

Commit suas mudanças (git commit -m 'Add some AmazingFeature')

Push para a branch (git push origin feature/AmazingFeature)

Abra um Pull Request

📄 Licença
Este projeto é para fins educacionais. Distribuído sob a licença MIT.

✨ Autores
Seu Nome - Desenvolvimento e Documentação - @seu-github

🙏 Agradecimentos
Professor da disciplina de Processamento de Imagem e Visão Computacional

Comunidade OpenCV e scikit-learn

Kagle pelo dataset público

Documentação da Yoosee e contribuições da comunidade

<p align="center"> Desenvolvido com ❤️ para disciplina de Visão Computacional </p><p align="center"> <strong>🎥 Demonstração em Vídeo:</strong> <a href="#">Link para vídeo</a> </p> ```