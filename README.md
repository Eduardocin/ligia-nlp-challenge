# 🔍 Ligia NLP Challenge

> Solução completa de classificação de notícias (Real vs Fake) usando Machine Learning clássico e Deep Learning com otimização de hiperparâmetros.

Nota: Esta estrutura de projeto foi inicialmente gerada usando o template `cookiecutter-datascience`. Mantivemos a organização e convenções do template para facilitar reprodutibilidade, testes e contribuição.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Template: cookiecutter-datascience](https://img.shields.io/badge/template-cookiecutter--datascience-orange.svg)](https://drivendata.github.io/cookiecutter-data-science/)

## 📋 Descrição

Este projeto implementa pipelines completos de NLP para classificação binária de notícias, comparando abordagens clássicas (TF-IDF + ML tradicional) com técnicas modernas. O projeto inclui análise exploratória, pré-processamento avançado, modelagem baseline, otimização de hiperparâmetros e avaliação comparativa de modelos.

## Estrutura do Projeto

```
├── LICENSE                  <- Licença do projeto
├── Makefile                 <- Comandos de conveniência
├── README.md                <- Documentação principal
├── environment.yml          <- Ambiente conda completo
├── pyproject.toml           <- Configuração do projeto (black, ruff)
├── requirements.txt         <- Dependências com versões fixadas
│
├── data/
│   ├── raw/                  <- Dados originais imutáveis
│   │   ├── train.csv
│   │   └── test.csv
│   ├── interim/              <- Dados intermediários
│   └── processed/            <- Artefatos para modelagem
│       ├── train_clean.csv       <- Texto pré-processado (treino)
│       ├── test_clean.csv        <- Texto pré-processado (teste)
│       ├── X_train_tfidf.npz     <- Matriz TF-IDF (treino)
│       ├── X_val_tfidf.npz       <- Matriz TF-IDF (validação)
│       ├── X_test_tfidf.npz      <- Matriz TF-IDF (teste)
│       ├── tfidf_vectorizer.joblib
│       ├── y_train.csv / y_val.csv
│       ├── train_indices.csv / val_indices.csv
│       └── submission_*.csv      <- Arquivos de submissão gerados
│
├── models/
│   ├── mlclassico/           <- Modelos de ML clássico (scikit-learn / XGBoost)
│   │   ├── char-ngrams-baseline/  <- Baselines com Character N-grams
│   │   │   ├── model_0_xgboost.joblib
│   │   │   ├── model_1_extra_trees.joblib
│   │   │   ├── model_2_random_forest.joblib
│   │   │   ├── model_3_linearsvc.joblib
│   │   │   ├── model_4_sgdclassifier.joblib
│   │   │   └── metadata.json
│   │   └── optimized/             <- Modelos com hiperparâmetros otimizados
│   │       ├── extratrees_optimized.joblib
│   │       ├── linearsvc_optimized.joblib
│   │       ├── sgdclassifier_optimized.joblib
│   │       └── metadata.json
│   └── deeplearning/         <- Modelos de Deep Learning (TinyBERT)
│       └── tinybert_checkpoints/  <- Checkpoints HuggingFace Trainer
│
├── notebooks/               <- Jupyter notebooks (executar na ordem)
│   ├── 1.0-data-exploration.ipynb         <- EDA completa
│   ├── 2.0-preprocessing.ipynb            <- Pré-processamento e vetorização
│   ├── 3.0-baseline-models.ipynb          <- Baselines com 5 classificadores
│   ├── 3.1-hyperparameter-optimization.ipynb <- Otimização de hiperparâmetros
│   ├── 4.0-predictions.ipynb              <- Geração de submissões
│   └── 5.0-tinybert.ipynb                 <- Fine-tuning TinyBERT
│
├── reports/
│   ├── baseline_results.csv
│   ├── baseline_results_char_ngrams.csv
│   ├── optimized_results.csv
│   ├── preprocessing_experiments_results.csv
│   └── figures/              <- Gráficos e figuras gerados pelos notebooks
│
├── references/              <- Referências e materiais de apoio
│
└── src/
    └── __init__.py
```

## ⚙️ Configuração do Ambiente

### Opção 1: Conda (Recomendado - Reprodutibilidade Garantida)

```bash
# Clonar repositório
git clone https://github.com/seu-usuario/ligia-nlp-challenge.git
cd ligia-nlp-challenge

# Criar ambiente conda com todas as dependências
conda env create -f environment.yml

# Ativar ambiente
conda activate ligia-nlp

# Verificar instalação
python -c "import sklearn, xgboost, transformers; print('✅ Ambiente configurado com sucesso!')"

# Iniciar Jupyter Lab
jupyter lab
```

**Para atualizar o ambiente existente:**
```bash
conda env update -f environment.yml --prune
```

## 🗂️ Notebooks — Pipeline de Análise

Execute os notebooks **na ordem abaixo** para reproduzir o pipeline completo.

### 1. Análise Exploratória de Dados (EDA)
[notebooks/1.0-data-exploration.ipynb](notebooks/1.0-data-exploration.ipynb)
- Estatísticas descritivas do dataset
- Distribuição de classes (balanceamento)
- Wordclouds e n-gramas por classe
- Identificação de data leakage (coluna `subject`)
- Análise de comprimento textual e padrões temporais

### 2. Pré-processamento de Texto
[notebooks/2.0-preprocessing.ipynb](notebooks/2.0-preprocessing.ipynb)
- Limpeza de texto (URLs, HTML, caracteres especiais)
- Remoção de duplicatas
- Experimentos com diferentes estratégias de normalização
- Vetorização TF-IDF com Character N-grams (4-6)
- Exporta: `data/processed/train_clean.csv`, `X_train_tfidf.npz`, `tfidf_vectorizer.joblib`

### 3. Modelos Baseline
[notebooks/3.0-baseline-models.ipynb](notebooks/3.0-baseline-models.ipynb)
- 5 classificadores com 5-Fold Cross-Validation:
  - Random Forest, Extra Trees, Logistic Regression, LinearSVC, SGDClassifier
- Avaliação: Accuracy, F1 Weighted, Matriz de Confusão
- Análise de erros e top-features (Character N-grams)
- Salva modelos em `models/mlclassico/char-ngrams-baseline/`

### 3.1. Otimização de Hiperparâmetros
[notebooks/3.1-hyperparameter-optimization.ipynb](notebooks/3.1-hyperparameter-optimization.ipynb)
- `RandomizedSearchCV` para LinearSVC, SGDClassifier e ExtraTrees
- Comparação antes/depois da otimização
- Salva modelos em `models/mlclassico/optimized/`

### 4. Geração de Predições e Submissão
[notebooks/4.0-predictions.ipynb](notebooks/4.0-predictions.ipynb)
- Carrega modelo XGBoost + vetorizador TF-IDF
- Gera arquivo de submissão para Kaggle
- Análise de concordância entre modelos (XGBoost vs ExtraTrees)
- Ensemble por votação (XGBoost + LinearSVC)

### 5. Fine-tuning TinyBERT
[notebooks/5.0-tinybert.ipynb](notebooks/5.0-tinybert.ipynb)
- Modelo: `huawei-noah/TinyBERT_General_4L_312D` (~14.5 M parâmetros)
- Tokenização WordPiece com `title [SEP] text`
- HuggingFace Trainer com cosine schedule, label smoothing e early stopping
- Checkpoints salvos em `models/deeplearning/tinybert_checkpoints/`
- Geração de submissão em `data/processed/submission_tinybert.csv`
- > **Nota:** Projetado para rodar com GPU. Em CPU o treinamento é muito lento.

---

## 🚀 Como Usar

### Reproduzir Pipeline Completo

```bash
# 1. Clonar repositório
git clone https://github.com/seu-usuario/ligia-nlp-challenge.git
cd ligia-nlp-challenge

# 2. Configurar ambiente (Conda — recomendado)
conda env create -f environment.yml
conda activate ligia-nlp

# Alternativa: pip
pip install -r requirements.txt

# 3. Verificar instalação
python -c "import sklearn, xgboost, transformers; print('Ambiente OK')"

# 4. Iniciar Jupyter Lab e executar notebooks na ordem
jupyter lab
```

**Ordem de execução obrigatória:**

| # | Notebook | Tempo estimado |
|---|----------|----------------|
| 1 | `1.0-data-exploration.ipynb` | ~5 min |
| 2 | `2.0-preprocessing.ipynb` | ~10 min |
| 3 | `3.0-baseline-models.ipynb` | ~15 min |
| 4 | `3.1-hyperparameter-optimization.ipynb` | ~45-60 min |
| 5 | `4.0-predictions.ipynb` | ~2 min |
| 6 | `5.0-tinybert.ipynb` | ~10 min (GPU) |

### Submeter no Kaggle

Após executar o notebook `4.0-predictions.ipynb`, os arquivos de submissão estarão em `data/processed/`:

```bash
# Submeter via Kaggle CLI
kaggle competitions submit -c <competition-name> \
  -f data/processed/submission_xgboost.csv \
  -m "XGBoost baseline - char ngrams"
```

### Usar Modelo Pré-treinado

```python
import joblib
import scipy.sparse

# Carregar vetorizador e modelo otimizado
tfidf = joblib.load('data/processed/tfidf_vectorizer.joblib')
model = joblib.load('models/mlclassico/char-ngrams-baseline/model_0_xgboost.joblib')

# Fazer predições
texts = ["Breaking: President signs new bill into law"]
X = tfidf.transform(texts)
predictions = model.predict(X)
print(f"Predição: {'Fake' if predictions[0] == 1 else 'Real'}")

# Modelo otimizado (LinearSVC)
model_opt = joblib.load('models/mlclassico/optimized/linearsvc_optimized.joblib')
```

## ️ Principais Tecnologias

| Categoria | Bibliotecas |
|-----------|-------------|
| Dados | `pandas`, `numpy`, `scipy` |
| Visualização | `matplotlib`, `seaborn`, `wordcloud` |
| NLP | `nltk` |
| ML Clássico | `scikit-learn`, `xgboost` |
| Deep Learning | `torch`, `transformers`, `datasets`, `accelerate` |
| Serialização | `joblib` |
| Ambiente | Jupyter Lab, conda |

## 📂 Dados

- **Fonte:** Dataset de notícias reais e falsas (Kaggle — LIGIA NLP Challenge)
- **Treino:** `data/raw/train.csv` (~22.8k amostras)
- **Teste:** `data/raw/test.csv` (~5k amostras)
- **Colunas:** `id`, `title`, `text`, `subject`, `date`, `label` (0 = Real, 1 = Fake)
- **Balanceamento:** Leve desbalanceamento

> ⚠️ **Data Leakage:** A coluna `subject` é um proxy perfeito do label e deve ser **descartada** na modelagem.

## 📝 Licença

Este projeto está sob a licença especificada no arquivo [LICENSE](LICENSE).

## 👤 Autor

**Eduardo** - [GitHub](https://github.com/seu-usuario)

---

⭐ Se este projeto foi útil, considere dar uma estrela!
