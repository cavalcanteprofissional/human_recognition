# TODO - Human Recognition Project

## Modificações Realizadas

### 1. Poetry: Orquestração de dependências
**pyproject.toml** — atualizado com todas as deps do projeto

| Dependência | Status | Motivo |
|---|---|---|
| `streamlit` | ✅ adicionado | estava só no `requirements.txt` |
| `av` (PyAV) | ✅ adicionado | proxy RTSP Yoosee |
| `seaborn` | ✅ adicionado | estava só no `requirements.txt` |
| `kaggle` | ✅ adicionado | `data_loader.py` usava (`ModuleNotFoundError`) |
| `opencv-python-headless` | ❌ substituído | não tinha `cv2.imshow()` |
| `opencv-python` | ✅ adicionado | GUI para detecção no terminal |

Outras mudanças:
- `package-mode = false` (projeto é aplicação, não pacote)
- `[tool.poetry.scripts]` com `hr` e `hr-dashboard`
- Python `^3.10`

### 2. Lazy imports em `run.py`
**Arquivo:** `run.py`

| Linha | Mudança |
|---|---|
| `12-15` | Removidos imports top-level de `data_loader`, `train`, `real_time_detector`, `utils` |
| `17` | `HumanDatasetLoader` movido para dentro de `setup_project()` |
| `46` | `HumanDatasetLoader` adicionado em `run_advanced_training()` |
| `222` | `train_main` movido para dentro de `if args.train:` |
| `256-257` | `plot_training_results` movido para `if args.analyze:` |
| `260-261` | `create_sample_comparison` movido para `if args.compare_filters:` |

**Motivo:** Evitar erro de autenticação Kaggle ao executar `--detect`, `--dashboard`, `--help`.

### 3. Paths relativos em `splits.npz`
**Arquivo:** `src/data_loader.py`

| Linha | Antes | Depois |
|---|---|---|
| `155` | `str(img_path)` (absoluto) | `str(img_path.relative_to(self.processed_path))` |
| `213-219` | retornava paths direto | `resolve_paths()` converte relativo → absoluto |

**Motivo:** `splits.npz` armazenava caminhos absolutos da unidade `E:\`, mas o projeto foi movido para `D:\`. Paths relativos tornam o splits portátil.

### 4. `splits.npz` regenerado
Script auxiliar: `tools/regenerate_splits.py`

```
Treino: 644 | Val: 138 | Teste: 139
```

### 5. opencv-python (com GUI) substitui headless
**pyproject.toml:10** — `opencv-python-headless` → `opencv-python`

**Motivo:** `cv2.imshow()` e `cv2.waitKey()` não existem na versão headless. Com `opencv-python` funciona tanto no terminal quanto no dashboard.

### 6. Modelo antigo removido (causava `_loss` error)
`models/best_model_20260223_203343.pkl` → movido para `.bak`

**Motivo:** Foi salvo com versão antiga do lightgbm. O módulo C `_loss` não é encontrado com a versão atual do lightgbm/xgboost.

### 7. Modelo treinado
`models/model_20260613_135702.pkl` — Random Forest com GridSearchCV.

---

## Pendências Resolvidas

### P1 — Erro `No module named '_loss'`
**Causa:** Modelo antigo `best_model_20260223_203343.pkl` foi salvo com uma versão do lightgbm que usava o módulo C `_loss`. O lightgbm e xgboost atuais importam sem erro.

**Solução:** Modelo antigo movido para `models/best_model_20260223_203343.pkl.bak`. O modelo novo (`model_20260613_135702.pkl` - RandomForest) carrega perfeitamente.

## Pendências

### P1 — Kaggle authentication blocking imports
O `import kaggle` no topo de `data_loader.py` exige autenticação mesmo para comandos que não usam Kaggle. Já contornado com lazy imports em `run.py`, mas o ideal seria isolar `kaggle` em um sub-módulo.
