# Human Recognition Project

<p align="center">
  <img src="https://raw.githubusercontent.com/opencv/opencv/master/doc/opencv-logo.png" width="300" alt="OpenCV Logo"/>
</p>

<p align="center">
  <strong>Projeto de Visão Computacional para reconhecimento de silhueta humana em tempo real</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/OpenCV-4.8+-green.svg" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/Streamlit-1.28+-red.svg" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/>
</p>

---

## 📋 Sobre o Projeto

Este projeto implementa um sistema completo de reconhecimento de silhueta humana utilizando técnicas clássicas de Visão Computacional e Machine Learning. O sistema é capaz de:

- Treinar um classificador Random Forest do zero usando características LBP (Local Binary Patterns)
- Detectar presença humana em tempo real via webcam ou câmera IP Yoosee
- Aplicar 6 filtros criativos diferentes para visualização estilizada
- Visualizar métricas e resultados em um dashboard interativo com Streamlit

---

## 🎯 Objetivos Acadêmicos

Este projeto foi desenvolvido como Trabalho Final para a disciplina de Processamento de Imagem e Visão Computacional, atendendo aos seguintes requisitos:

- ✅ Implementação de algoritmo do zero (não usar soluções prontas)
- ✅ Dataset público e bem documentado
- ✅ Extração manual de características (LBP)
- ✅ Treinamento com variação de hiperparâmetros
- ✅ Aplicação em tempo real com webcam/câmera IP
- ✅ Dashboard interativo com métricas e visualizações
- ✅ Documentação completa do pipeline

---

## 🚀 Começando

### Pré-requisitos
- Python 3.9+
- Poetry (gerenciador de dependências)
- Webcam ou Câmera IP Yoosee
- Conta no Kaggle (para download do dataset)
- PyAV (`pip install av`) - para suporte a autenticação Digest RTSP
- XGBoost e LightGBM (`pip install xgboost lightgbm`) - para modelos avançados

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/human_recognition.git
cd human_recognition
```

2. Instale as dependências com Poetry:
```bash
poetry install
```

3. Ative o ambiente virtual:
```bash
poetry shell
```

4. Configure as variáveis de ambiente:

Crie um arquivo `.env` na raiz do projeto:

```env
# Credenciais Kaggle (obrigatório para download do dataset)
KAGGLE_USERNAME=seu_usuario_kaggle
KAGGLE_KEY=sua_chave_kaggle

# Configurações da Câmera Yoosee
YOOSEE_IP=192.168.100.49
YOOSEE_PORT=554
YOOSEE_USERNAME=admin
YOOSEE_PASSWORD=HonkaiImpact3rd
YOOSEE_STREAM=onvif1
```

---

## 📦 Estrutura do Projeto

```
human_recognition/
├── .env                      # Variáveis de ambiente
├── .gitignore                # Arquivos ignorados pelo git
├── pyproject.toml            # Dependências do Poetry
├── README.md                 # Este arquivo
├── AGENTS.md                 # Instruções para agentes
├── run.py                    # Script principal
│
├── data/                     # Dados do projeto
│   ├── raw/                  # Dataset original
│   └── processed/            # Dados processados
│
├── models/                   # Modelos treinados
│   └── model_*.pkl
│
├── reports/                  # Relatórios e figuras
│
├── src/                      # Código fonte
│   ├── __init__.py
│   ├── config.py             # Configurações
│   ├── data_loader.py        # Carregamento do dataset
│   ├── feature_extractor.py  # Extração LBP
│   ├── train.py              # Treinamento básico (RF)
│   ├── train_advanced.py     # Treinamento avançado (múltiplos modelos)
│   ├── model_registry.py     # Registro de modelos
│   ├── ensemble.py           # Voting/Stacking ensembles
│   ├── real_time_detector.py # Detecção em tempo real
│   ├── yoosee_camera.py     # Integração com câmera Yoosee
│   └── utils.py              # Utilitários
│
└── tools/                    # Ferramentas auxiliares
    ├── find_yoosee_ip.py    # Scanner para encontrar câmera
    ├── test_yoosee_connection.py
    ├── test_digest_auth.py
    ├── rtsp_client.py       # Cliente RTSP com Digest Auth
    ├── rtsp_to_mjpeg.py     # Proxy RTSP→HTTP (PyAV)
    ├── rtsp_gateway.py     # Gateway FFmpeg
    └── yoosee_proxy.py
```

---

## 🎯 Funcionalidades

### 1. Pipeline de Machine Learning
- **Dataset**: Human Detection Dataset (Kaggle) com 921 imagens 256x256
- **Divisão**: 70% treino / 15% validação / 15% teste
- **Extração de características**: LBP (Local Binary Patterns) com 59 features
- **Validação Cruzada**: 5-fold CV com Grid Search de hiperparâmetros
- **Métricas**: Acurácia, Precisão, Recall, F1-Score, AUC-ROC, Matriz de Confusão

### 2. Modelos Disponíveis (8 classificadores)

| Modelo | Tipo | Descrição |
|--------|------|-----------|
| Random Forest | Ensemble (Bagging) | Floresta aleatória |
| Gradient Boosting | Ensemble (Boosting) | Boosting sequencial |
| XGBoost | Ensemble (Boosting) | Extreme Gradient Boosting |
| LightGBM | Ensemble (Boosting) | Light Gradient Boosting |
| SVM | Kernel | Support Vector Machine (RBF) |
| KNN | Instance-based | K-Nearest Neighbors |
| Logistic Regression | Linear | Regressão logística |
| MLP | Neural Network | Perceptron multicamadas |

### 3. Ensemble de Modelos
- **Voting Ensemble**: Combina predições dos melhores modelos
- **Seleção automática**: Os 5 melhores modelos formam o ensemble

### 4. Detecção em Tempo Real
- **Webcam local**: Suporte nativo via OpenCV
- **Câmera Yoosee**: Integração via RTSP/ONVIF com autenticação Digest via proxy PyAV
- **Baixa latência**: Streaming otimizado para tempo real
- **IP Dinâmico**: Auto-discovery na rede local

### 5. Filtros Criativos

| Filtro | Descrição |
|--------|-----------|
| cartoon | Efeito cartoon com bordas suaves |
| edges | Detecção de bordas coloridas (Canny) |
| colormap | Mapas de cor criativos (OCEAN, JET) |
| stylized | Efeito artístico estilizado |
| pencil | Efeito de desenho a lápis |
| none | Sem filtro |

### 6. Dashboard Interativo
- **Visão Geral**: Pipeline completo e explicação do LBP
- **Treinamento**: Configuração de parâmetros e grid search
- **Detecção**: Transmissão ao vivo com estatísticas
- **Análise**: Gráficos interativos e matriz de confusão

---

## 🎮 Como Usar

### 1. Setup Inicial
```bash
poetry run python run.py --setup
```

### 2. Treinar Modelo

#### Treinamento Básico (Random Forest)
```bash
poetry run python run.py --train
```

#### Treinamento Avançado (Múltiplos Modelos)
```bash
# Treinar todos os 8 modelos + ensemble
poetry run python run.py --train-advanced

# Treinar modelos específicos
poetry run python run.py --train-advanced --models random_forest,xgboost,svm

# Com mais folds de validação cruzada
poetry run python run.py --train-advanced --cv-folds 10

# Sem ensemble
poetry run python run.py --train-advanced --no-ensemble

# Listar modelos disponíveis
poetry run python run.py --list-models

# Comparar resultados de treinamentos anteriores
poetry run python run.py --compare-models
```

### 3. Executar Dashboard
```bash
poetry run python run.py --dashboard
```
Acesse: http://localhost:8501

### 4. Detecção em Tempo Real
```bash
# Webcam com filtro cartoon
poetry run python run.py --detect

# Com filtro específico
poetry run python run.py --detect --filter edges
poetry run python run.py --detect --filter colormap
```

### 5. Análise de Resultados
```bash
poetry run python run.py --analyze reports/results_*.json
```

---

## 📹 Integração com Câmera Yoosee

### IP Dinâmico
```bash
# Buscar câmera automaticamente
poetry run python run.py --auto-find-yoosee
```

### Detecção com Auto-Discovery
```bash
poetry run python run.py --detect --source yoosee --auto-find-yoosee
```

### Detecção com IP Fixo
```bash
python run.py --detect --source yoosee --yoosee-ip 192.168.100.49
```

### Teste de Conexão
```bash
python tools/test_yoosee_connection.py --ip 192.168.100.49 --diagnose
```

---

## 🔧 Solução de Problemas

### Câmera Yoosee não conecta

1. **IP dinâmico** (recomendado):
```bash
python run.py --auto-find-yoosee
```
O sistema escaneia a rede automaticamente e atualiza o arquivo `.env`.

2. **Problemas de autenticação**:
- Verifique no app Yoosee se RTSP está habilitado
- Confirme a senha correta
- O modelo LB-CA128 requer autenticação Digest (suportada automaticamente)

3. **Proxy PyAV**:
Se o FFmpeg falhar (erro "Nonmatching transport"), o sistema usa automaticamente o proxy PyAV que implementa autenticação Digest nativamente.

4. **Teste de diagnóstico**:
```bash
python tools/test_yoosee_connection.py --ip 192.168.100.49 --diagnose
```

### Teste Manual do Proxy

```bash
# Iniciar proxy
python tools/rtsp_to_mjpeg.py --preview

# Ou via API
python tools/rtsp_to_mjpeg.py --ip 192.168.100.49 --user admin --password HonkaiImpact3rd
```

### Modelos Yoosee e Caminhos RTSP

| Modelo | Caminho |
|--------|---------|
| C100E | /onvif1 |
| J1080P | /onvif1, /onvif2 |
| LB-CA128 | /onvif1 (Digest) |

---

## 📊 Resultados Esperados

- **Acurácia**: > 85%
- **FPS (Webcam)**: ~30 FPS
- **FPS (Yoosee)**: ~15-20 FPS
- **Latência**: < 100ms

---

## 📄 Licença

Este projeto é para fins educacionais. Distribuído sob a licença MIT.

---

<p align="center">
  Desenvolvido para disciplina de Visão Computacional
</p>
