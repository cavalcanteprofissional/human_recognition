# Human Recognition Project

Projeto de Visão Computacional para reconhecimento de silhueta humana em tempo real utilizando LBP (Local Binary Patterns) e Random Forest.

## 🚀 Configuração Rápida

### Pré-requisitos
- Python 3.9+
- Poetry
- Webcam

### Instalação

1. Clone o repositório:
```bash
git clone <seu-repositorio>
cd human_recognition
Instale as dependências com Poetry:

bash
poetry install
Configure as credenciais do Kaggle no arquivo .env:

env
KAGGLE_USERNAME=cavalcantesantos
KAGGLE_KEY=ae9786f4a28869eeef14490073738a3c
Ative o ambiente virtual:

bash
poetry shell
📦 Comandos
Setup Inicial
Baixa e prepara o dataset:

bash
python run.py --setup
Treinamento
Treina o modelo com busca de hiperparâmetros:

bash
python run.py --train
Dashboard Interativo (Recomendado)
Executa o dashboard Streamlit com todas as funcionalidades:

bash
# Via run.py
python run.py --dashboard

# Ou diretamente
python run_dashboard.py
O dashboard estará disponível em: http://localhost:8501

Detecção em Tempo Real (Terminal)
Executa o detector com diferentes filtros:

bash
# Com filtro cartoon (padrão)
python run.py --detect

# Com filtro específico
python run.py --detect --filter edges
python run.py --detect --filter colormap
python run.py --detect --filter stylized
python run.py --detect --filter pencil
python run.py --detect --filter none
Análise
Visualiza resultados do treinamento:

bash
python run.py --analyze reports/results_20240101_120000.json
Compara todos os filtros:

bash
python run.py --compare-filters
📊 Dashboard Interativo
O dashboard Streamlit oferece 5 abas principais:

1. Visão Geral
Pipeline completo do projeto

Explicação didática do LBP

Cards com informações principais

2. Treinamento
Configuração de parâmetros

Visualização dos resultados da busca em grade

Heatmap interativo de parâmetros

Top 10 configurações

3. Detecção em Tempo Real
Transmissão ao vivo da webcam

Seleção de filtros em tempo real

Métricas atualizadas (classe, confiança)

Histórico das últimas detecções

4. Análise de Métricas
Gráfico de radar com todas as métricas

Matriz de confusão interativa

Curvas de aprendizado

Comparação de parâmetros

5. Sobre o Projeto
Descrição detalhada

Informações do dataset

Explicação completa do LBP

Galeria de filtros

🎨 Filtros Disponíveis
cartoon: Efeito cartoon com bordas suaves

edges: Detecção de bordas coloridas

colormap: Mapas de cor criativos (OCEAN, JET, etc.)

stylized: Efeito artístico estilizado

pencil: Efeito de desenho a lápis

none: Sem filtro

📁 Estrutura do Projeto
text
human_recognition/
├── data/               # Dados brutos e processados
├── models/             # Modelos treinados
├── reports/            # Relatórios e figuras
├── src/                # Código fonte
│   ├── config.py       # Configurações
│   ├── data_loader.py  # Carregamento de dados
│   ├── feature_extractor.py  # Extração LBP
│   ├── train.py        # Treinamento
│   ├── real_time_detector.py # Detecção em tempo real
│   ├── dashboard.py    # Dashboard Streamlit
│   └── utils.py        # Utilitários
├── .env                # Credenciais
├── pyproject.toml      # Dependências
├── run.py              # Script principal
└── run_dashboard.py    # Script do dashboard
📊 Métricas
O projeto calcula as seguintes métricas:

Acurácia: (VP + VN) / (VP + VN + FP + FN)

Precisão: VP / (VP + FP)

Revocação (Recall): VP / (VP + FN)

F1-Score: 2 * (Precisão * Recall) / (Precisão + Recall)

Matriz de Confusão: VP, VN, FP, FN

Onde:

VP = Verdadeiros Positivos (humano detectado corretamente)

VN = Verdadeiros Negativos (não humano detectado corretamente)

FP = Falsos Positivos (falso alarme)

FN = Falsos Negativos (humano não detectado)

🎯 Funcionalidades do Dashboard
Visualizações Interativas
Heatmaps interativos para correlação de parâmetros

Gráficos de radar para comparação de métricas

Matriz de confusão com Plotly

Curvas de aprendizado dinâmicas

Detecção em Tempo Real
Transmissão ao vivo com baixa latência

Troca de filtros em tempo real

Estatísticas atualizadas automaticamente

Histórico de detecções

Análise de Modelos
Comparação de múltiplos modelos

Visualização de hiperparâmetros

Exportação de resultados

📝 Licença
Este projeto é para fins educacionais.

text

## Como Executar o Dashboard

1. **Ative o ambiente:**
```bash
poetry shell
Execute o dashboard:

bash
# Opção 1: Via run.py
python run.py --dashboard

# Opção 2: Script dedicado
python run_dashboard.py
Acesse no navegador:

text
http://localhost:8501
Características do Dashboard
O dashboard Streamlit oferece:

Interface moderna e responsiva com CSS personalizado

Visualizações interativas com Plotly

Detecção em tempo real integrada

Métricas atualizadas automaticamente

Seleção de modelos treinados

Configuração de parâmetros em tempo real

Histórico de detecções com DataFrame

Explicações didáticas do LBP e do pipeline