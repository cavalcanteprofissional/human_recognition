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

- Treinar múltiplos classificadores usando características LBP (Local Binary Patterns)
- Detectar presença humana em tempo real via webcam ou câmera IP Yoosee
- Aplicar 6 filtros criativos diferentes para visualização estilizada
- Visualizar métricas detalhadas em dashboard interativo com Streamlit
- Processamento paralelo para otimização de performance

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
- ✅ Paralelização para otimização de performance

---

## 📦 Origem do Dataset

### Human Detection Dataset

**Fonte:** [Kaggle - constantinwerner/human-detection-dataset](https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset)

### Descrição

O dataset utilizado neste projeto é o **Human Detection Dataset**, um conjunto de imagens públicas desenvolvido para tarefas de classificação binária de detecção humana. O dataset foi escolhido por sua adequação aos requisitos acadêmicos do projeto:

| Característica | Descrição |
|----------------|-----------|
| **Total de Imagens** | 921 imagens |
| **Resolução** | 256 × 256 pixels |
| **Formato** | PNG (escala de cinza e RGB) |
| **Classes** | 2 (binário) |
| **Licença** | CC0 - Domínio Público |

### Estrutura de Classes

```
human-detection-dataset/
├── 0/                    # Classe: Sem Humano
│   ├── image_001.png
│   ├── image_002.png
│   └── ... (imagens de cenas vazias, objetos, backgrounds)
│
└── 1/                    # Classe: Com Humano
    ├── image_001.png
    ├── image_002.png
    └── ... (imagens contendo silhuetas/pessoas)
```

| Classe | Descrição | Exemplos |
|--------|-----------|----------|
| **0 (no_human)** | Cenas sem presença humana | Ambientes vazios, objetos isolados, paisagens, interiores |
| **1 (human)** | Cenas com presença humana | Pessoas completas, silhuetas, grupos |

### Distribuição dos Dados

O dataset é dividido utilizando estratificação para manter o balanceamento:

| Conjunto | Proporção | Quantidade (~) |
|----------|-----------|----------------|
| Treino | 70% | ~645 imagens |
| Validação | 15% | ~138 imagens |
| Teste | 15% | ~138 imagens |

### Justificativa de Escolha

1. **Dataset Público**: Disponível no Kaggle, sem restrições de uso acadêmico
2. **Bem Documentado**: Metadados claros sobre coleta e anotação
3. **Tamanho Adequado**: Suficiente para treinamento, pequeno para iteração rápida
4. **Balanceamento**: Distribuição aproximadamente equilibrada entre classes
5. **Variabilidade**: Diversidade de cenários, iluminação e poses

---

## 🚀 Começando

### Pré-requisitos
- Python 3.9+
- Webcam ou Câmera IP Yoosee
- Conta no Kaggle (para download do dataset)

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/human_recognition.git
cd human_recognition
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:

Crie um arquivo `.env` na raiz do projeto:

```env
# Credenciais Kaggle (obrigatório para download do dataset)
KAGGLE_USERNAME=seu_usuario_kaggle
KAGGLE_KEY=sua_chave_kaggle

# Configurações da Câmera Yoosee
YOOSEE_IP=192.168.100.49
YOOSEE_PORT=554
YOOSEE_USERNAME=admin
YOOSEE_PASSWORD=sua_senha
YOOSEE_STREAM=onvif1
```

---

## 📂 Estrutura do Projeto

```
human_recognition/
├── .env                      # Variáveis de ambiente
├── .gitignore                # Arquivos ignorados pelo git
├── requirements.txt          # Dependências Python
├── README.md                 # Este arquivo
├── AGENTS.md                 # Instruções para agentes
├── LICENSE                   # Licença MIT
├── run.py                    # Script principal
├── dashboard.py              # Dashboard Streamlit (5 tabs)
│
├── data/                     # Dados do projeto
│   ├── raw/                  # Dataset original
│   └── processed/            # Dados processados
│
├── models/                   # Modelos treinados
│   ├── model_*.pkl           # Modelos básicos
│   └── best_model_*.pkl      # Melhor modelo
│
├── reports/                  # Relatórios e figuras
│   ├── model_comparison_*.json  # Resultados com métricas por fold
│   └── *.png                 # Gráficos e matrizes de confusão
│
├── src/                      # Código fonte
│   ├── __init__.py
│   ├── config.py             # Configurações
│   ├── data_loader.py        # Carregamento do dataset
│   ├── feature_extractor.py  # Extração LBP (paralelizado)
│   ├── train.py              # Treinamento básico (RF) + GridSearchCV
│   ├── train_advanced.py     # Treinamento avançado (paralelizado)
│   ├── model_registry.py     # Registro de modelos + cv_fold_metrics
│   ├── ensemble.py           # Voting/Stacking ensembles
│   ├── real_time_detector.py # Detecção em tempo real
│   ├── yoosee_camera.py      # Integração com câmera Yoosee
│   └── utils.py              # Utilitários
│
└── tools/                    # Ferramentas auxiliares
    ├── find_yoosee_ip.py     # Scanner para encontrar câmera
    ├── test_yoosee_connection.py
    ├── test_digest_auth.py
    ├── rtsp_client.py        # Cliente RTSP com Digest Auth
    ├── rtsp_to_mjpeg.py      # Proxy RTSP→HTTP (PyAV)
    └── rtsp_gateway.py       # Gateway FFmpeg
```

---

## 🎯 Funcionalidades

### 1. Pipeline de Machine Learning
- **Dataset**: Human Detection Dataset (Kaggle) com 921 imagens 256x256
- **Divisão**: 70% treino / 15% validação / 15% teste
- **Extração de características**: LBP (Local Binary Patterns) com 59 features
- **Validação Cruzada**: 5-fold CV com Grid Search de hiperparâmetros
- **Métricas**: Acurácia, Precisão, Recall, F1-Score, AUC-ROC, Matriz de Confusão
- **Métricas por Fold**: Captura detalhada de cada fold da validação cruzada

### 2. Paralelização (Otimização de Performance)

| Componente | Implementação |
|------------|---------------|
| Extração LBP | `joblib.Parallel` com todos os cores |
| Treinamento de Modelos | Modelos paralelos com `joblib.Parallel` |
| Grid Search | `GridSearchCV` com `n_jobs=-1` |
| Random Forest | `n_jobs=-1` interno |

### 3. Modelos Disponíveis (8 classificadores)

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

### 4. Ensemble de Modelos
- **Voting Ensemble**: Combina predições dos melhores modelos
- **Seleção automática**: Os 5 melhores modelos formam o ensemble
- **Soft Voting**: Usa probabilidades para decisão

### 5. Detecção em Tempo Real
- **Webcam local**: Suporte nativo via OpenCV
- **Câmera Yoosee**: Integração via RTSP/ONVIF com autenticação Digest via proxy PyAV
- **Baixa latência**: Streaming otimizado para tempo real
- **IP Dinâmico**: Auto-discovery na rede local

### 6. Filtros Criativos

| Filtro | Descrição |
|--------|-----------|
| cartoon | Efeito cartoon com bordas suaves |
| edges | Detecção de bordas coloridas (Canny) |
| colormap | Mapas de cor criativos (OCEAN, JET) |
| stylized | Efeito artístico estilizado |
| pencil | Efeito de desenho a lápis |
| none | Sem filtro |

---

## 📊 Dashboard Interativo

O dashboard foi desenvolvido com **Streamlit** e possui **5 abas (tabs)** para navegação organizada das funcionalidades. Acesse via:

```bash
streamlit run dashboard.py
```

### Tab 1: 📊 Métricas Gerais

**Objetivo:** Visão consolidada e comparativa de todos os modelos treinados.

| Elemento | Descrição |
|----------|-----------|
| **Seletor de Relatórios** | Lista todos os arquivos JSON em `/reports` ordenados por data |
| **Tabela Comparativa** | Exibe métricas de todos os modelos lado a lado |
| **Destaque do Melhor** | Indica o modelo com maior acurácia |
| **Download CSV** | Exporta tabela para análise externa |

**Métricas Exibidas:**

| Coluna | Descrição |
|--------|-----------|
| Modelo | Nome do classificador |
| Accuracy (CV) | Média da validação cruzada ± desvio padrão |
| Accuracy (Test) | Acurácia no conjunto de teste |
| Precision | Precisão (classe positiva) |
| Recall | Revocação (classe positiva) |
| F1-Score | Média harmônica Precision/Recall |
| Tempo (s) | Tempo de treinamento |

**Exemplo de Visualização:**
```
┌─────────────────┬──────────────┬───────────────┬───────────┐
│ Modelo          │ Acc (CV)     │ Acc (Test)    │ F1-Score  │
├─────────────────┼──────────────┼───────────────┼───────────┤
│ Random Forest   │ 0.85 ± 0.02  │ 0.87          │ 0.86      │
│ XGBoost         │ 0.84 ± 0.03  │ 0.86          │ 0.85      │
│ SVM             │ 0.83 ± 0.02  │ 0.84          │ 0.83      │
└─────────────────┴──────────────┴───────────────┴───────────┘
```

---

### Tab 2: 📈 Métricas por Fold

**Objetivo:** Análise detalhada fold-a-fold da validação cruzada.

| Elemento | Descrição |
|----------|-----------|
| **Seletor de Modelo** | Escolha qual modelo analisar |
| **Tabela por Fold** | Métricas individuais de cada fold |
| **Linha de Média** | Média das métricas across folds |
| **Linha de Std** | Desvio padrão das métricas |
| **Gráfico de Barras** | Visualização Accuracy e F1 por fold |

**Estrutura da Tabela:**

| Fold | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| 1 | 0.85 | 0.84 | 0.86 | 0.85 |
| 2 | 0.82 | 0.81 | 0.83 | 0.82 |
| 3 | 0.87 | 0.88 | 0.86 | 0.87 |
| 4 | 0.84 | 0.83 | 0.85 | 0.84 |
| 5 | 0.86 | 0.85 | 0.87 | 0.86 |
| **Média** | **0.848** | **0.842** | **0.854** | **0.848** |
| **Std** | **0.018** | **0.024** | **0.014** | **0.018** |

**Importante:** Esta tab requer treinamento via `--train-advanced` para gerar `cv_fold_metrics`.

---

### Tab 3: 🎥 Detecção em Tempo Real

**Objetivo:** Executar detecção ao vivo com webcam ou câmera Yoosee.

| Elemento | Descrição |
|----------|-----------|
| **Carregar Modelo** | Botão para carregar modelo treinado |
| **Stream de Vídeo** | Exibição do feed ao vivo com detecções |
| **Seletor de Filtro** | Escolha entre 6 filtros visuais |
| **Métricas Live** | Classe predita e confiança em tempo real |
| **Histórico** | Últimas 10 detecções realizadas |

**Fontes de Vídeo Disponíveis:**

| Fonte | Descrição | Requisito |
|-------|-----------|-----------|
| Webcam | Câmera local do computador | OpenCV |
| Yoosee | Câmera IP via RTSP/ONVIF | Configurar IP/senha |

**Filtros Disponíveis:**

| Filtro | Efeito Visual |
|--------|---------------|
| none | Imagem original |
| cartoon | Efeito cartoon com bordas realçadas |
| edges | Apenas bordas (Canny) |
| colormap | Mapa de cores (OCEAN) |
| stylized | Efeito artístico suave |
| pencil | Desenho a lápis |

**Fluxo de Uso:**
1. Clicar em "Carregar Modelo"
2. Escolher fonte (Webcam ou Yoosee)
3. Selecionar filtro desejado
4. Observar detecções em tempo real
5. Clicar "Parar" para encerrar

---

### Tab 4: 📉 Análise Visual

**Objetivo:** Visualizações gráficas dos resultados de treinamento.

| Elemento | Descrição |
|----------|-----------|
| **Gráfico Accuracy** | Barras comparando acurácia por modelo |
| **Gráfico F1-Score** | Barras comparando F1-Score por modelo |
| **Ranking** | Ordenação dos modelos por métrica |

**Tipos de Visualização:**

1. **Comparação de Acurácia:** Gráfico de barras horizontal com todos os modelos
2. **Comparação F1-Score:** Gráfico de barras horizontal ordenado
3. **Ranking por Métrica:** Lista ordenada para cada métrica disponível

**Exemplo de Ranking:**
```
🏆 Ranking por test_accuracy:
  1. random_forest
  2. xgboost
  3. gradient_boosting
  4. svm
  5. voting_ensemble
```

---

### Tab 5: ⚙️ Config/Sobre

**Objetivo:** Configurações do sistema e informações do projeto.

| Seção | Conteúdo |
|-------|----------|
| **Câmera Yoosee** | Formulário para configurar IP, usuário, senha e stream |
| **Dataset** | Contagem de imagens por classe |
| **Sobre** | Descrição geral do projeto |

**Configuração Yoosee:**

| Campo | Descrição |
|-------|-----------|
| IP | Endereço IP da câmera (ex: 192.168.100.49) |
| Usuário | Usuário para autenticação (padrão: admin) |
| Senha | Senha de acesso |
| Stream | Tipo de stream (onvif1, onvif2, live) |

**Status do Dataset:**
```
┌─────────────┬──────────┐
│ Classe      │ Imagens  │
├─────────────┼──────────┤
│ Humanos     │ 461      │
│ Não Humanos │ 460      │
└─────────────┴──────────┘
```

**Informações do Projeto:**
- **Features:** LBP (Local Binary Patterns)
- **Modelos:** Random Forest, XGBoost, SVM, KNN, etc.
- **Validação:** 5-fold Cross-Validation
- **Framework:** Streamlit + OpenCV + scikit-learn

---

## 🎮 Como Usar

### 1. Setup Inicial
```bash
python run.py --setup
```

### 2. Treinar Modelo

#### Treinamento Básico (Random Forest com GridSearchCV)
```bash
python run.py --train
```

#### Treinamento Avançado (Múltiplos Modelos)
```bash
# Treinar todos os 8 modelos + ensemble
python run.py --train-advanced

# Treinar modelos específicos
python run.py --train-advanced --models random_forest,xgboost,svm

# Com mais folds
python run.py --train-advanced --cv-folds 10

# Listar modelos disponíveis
python run.py --list-models

# Comparar resultados
python run.py --compare-models
```

### 3. Executar Dashboard
```bash
streamlit run dashboard.py
# ou
python run.py --dashboard
```

### 4. Detecção em Tempo Real
```bash
# Webcam
python run.py --detect

# Com filtro específico
python run.py --detect --filter edges

# Câmera Yoosee
python run.py --detect --source yoosee
```

---

## 📹 Integração com Câmera Yoosee

### Auto-Discovery
```bash
python run.py --auto-find-yoosee
```

### Detecção com Auto-Discovery
```bash
python run.py --detect --source yoosee --auto-find-yoosee
```

### Modelos Yoosee e Caminhos RTSP

| Modelo | Caminho |
|--------|---------|
| C100E | /onvif1 |
| J1080P | /onvif1, /onvif2 |
| LB-CA128 | /onvif1 (Digest Auth) |

---

## 🔧 Solução de Problemas

### Câmera Yoosee não conecta

```bash
# Buscar IP automaticamente
python run.py --auto-find-yoosee

# Teste de diagnóstico
python tools/test_yoosee_connection.py --ip 192.168.100.49 --diagnose
```

### Modelos não carregam
```bash
pip install xgboost lightgbm
```

### Métricas por fold não aparecem
```bash
# Treine novamente com treinamento avançado
python run.py --train-advanced
```

---

## 📊 Resultados Esperados

- **Acurácia**: > 85%
- **FPS (Webcam)**: ~30 FPS
- **FPS (Yoosee)**: ~15-20 FPS
- **Latência**: < 100ms
- **Speedup Paralelização**: 2-4x em máquinas multi-core

---

## 📄 Licença

Este projeto é para fins educacionais. Distribuído sob a licença MIT.

---

<p align="center">
  Desenvolvido para disciplina de Visão Computacional
</p>
