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
├── Makefile                 <- Comandos de conveniência (make data, make train)
├── README.md                <- Documentação principal para desenvolvedores
├── pyproject.toml           <- Configuração do projeto e ferramentas (black, ruff)
├── requirements.txt         <- Dependências para reprodução do ambiente
├── setup.cfg                <- Configuração de ferramentas de linting
│
├── data/
│   ├── external/             <- Dados de fontes externas
│   ├── interim/              <- Dados intermediários transformados
│   ├── processed/            <- Datasets finais para modelagem
│   └── raw/                  <- Dados originais imutáveis
│
├── docs/                    <- Documentação do projeto (mkdocs)
│
├── models/                  <- Modelos treinados, serializações e previsões
│   └── bert-baseline/        <- Checkpoints do modelo BERT fine-tuned
│
├── notebooks/               <- Jupyter notebooks para análise
│   ├── 1.0-initial-data-exploration.ipynb     <- EDA completa
│   ├── 2.0-text-preprocessing.ipynb           <- Limpeza de texto
│   ├── 3.0-baseline-models-tfidf.ipynb        <- Modelos baseline
│   ├── 3.1-hyperparameter-optimization.ipynb  <- Otimização XGBoost/LinearSVC
│   └── 4.0-prediction.ipynb                   <- Predições finais
│
├── references/              <- Dicionários de dados, manuais e materiais explicativos
│
├── reports/                 <- Análises geradas (HTML, PDF, LaTeX)
│   └── figures/              <- Gráficos e figuras para relatórios
│
└── src/                     <- Código-fonte do projeto
    ├── __init__.py           <- Torna src um módulo Python
    ├── config.py             <- Variáveis e configurações úteis
    ├── dataset.py            <- Scripts para download/geração de dados
    ├── features.py           <- Engenharia de features para modelagem
    ├── modeling/
    │   ├── __init__.py
    │   ├── predict.py        <- Inferência com modelos treinados
    │   └── train.py          <- Treinamento de modelos
    └── plots.py              <- Visualizações
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

## Notebooks

Lista de notebooks principais (sem descrições detalhadas):

- `notebooks/1.0-initial-data-exploration.ipynb`
- `notebooks/2.0-text-preprocessing.ipynb`
- `notebooks/3.0-baseline-models-tfidf.ipynb`
- `notebooks/3.1-hyperparameter-optimization.ipynb`
- `notebooks/4.0-prediction.ipynb`

## Pipeline de Análise

### 1. Análise Exploratória de Dados (EDA)
[notebooks/1.0-initial-data-exploration.ipynb](notebooks/1.0-initial-data-exploration.ipynb)
- Estatísticas descritivas
- Análise de distribuição de classes
- Visualização de padrões textuais (wordclouds, n-gramas)
- Identificação de data leakage

### 2. Pré-processamento de Texto
[notebooks/2.0-text-preprocessing.ipynb](notebooks/2.0-text-preprocessing.ipynb)
- Remoção de URLs, menções, hashtags e HTML
- Tratamento de duplicatas
- Normalização de texto
- Exportação de datasets limpos

### 2.1. Pré-processamento Avançado com spaCy
[notebooks/2.1-spacy-text-preprocessing.ipynb](notebooks/2.1-spacy-text-preprocessing.ipynb)
- Lematização com `en_core_web_sm`
- Extração de features linguísticas (POS ratios, NER counts)
- Texto lematizado para uso em TF-IDF e modelos

### 3. Modelos Baseline (TF-IDF)
[notebooks/3.0-baseline-models-tfidf.ipynb](notebooks/3.0-baseline-models-tfidf.ipynb)
- Representação TF-IDF (unigramas + bigramas)
- Random Forest Classifier
- XGBoost Classifier
- Análise de erros e features importantes

### 3.1. Baseline spaCy BERT (Feature Extraction)
[notebooks/3.1-spacy-bert-baseline.ipynb](notebooks/3.1-spacy-bert-baseline.ipynb)
- Embeddings RoBERTa-base via `en_core_web_trf` (768-d)
- Logistic Regression + MLP sobre embeddings
- C🚀 Como Usar

### Reproduzir Pipeline Completo

```bash
# 1. Configurar ambiente
conda env create -f environment.yml
conda activate ligia-nlp

# 2. Executar notebooks na ordem
jupyter lab

# 3. Ordem de execução recomendada:
#    → 1.0-initial-data-exploration.ipynb
#    → 2.0-text-preprocessing.ipynb
#    → 3.0-baseline-models-tfidf.ipynb
#    → 3.1-hyperparameter-optimization.ipynb (otimização completa ~1h)
#    → 4.0-prediction.ipynb
```

### Usar Modelo Pré-treinado

```python
import joblib
import pandas as pd

# Carregar modelo e vetorizador
model = joblib.load('models/optimized/xgboost_optimized.joblib')
tfidf = joblib.load('models/optimized/tfidf_vectorizer.joblib')

# Fazer predições
texts = ["Exemplo de notícia para classificar"]
X = tfidf.transform(texts)
predictions = model.predict(X)
print(f"Predição: {'Fake' if predictions[0] == 1 else 'Real'}")
```

## 📂 Dados

- **Fonte:** Dataset de notícias reais e falsas
- **Train:** `data/raw/train.csv` (~20k amostras)
- **Test:** `data/raw/test.csv` (~5k amostras)
- **Colunas:** `title`, `text`, `subject`, `label` (0=Real, 1=Fake)

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:
1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença especificada no arquivo [LICENSE](LICENSE).

## 👤 Autor

**Eduardo** - [GitHub](https://github.com/seu-usuario)

---

⭐ Se este projeto foi útil, considere dar uma estrela!

## Principais Tecnologias

- **Processamento de Dados:** pandas, numpy
- **Visualização:** matplotlib, seaborn, wordcloud
- **NLP:** nltk
- **Machine Learning:** scikit-learn, xgboost
- **Otimização:** RandomizedSearchCV, GridSearchCV
- **Ambiente:** Jupyter Lab/Notebook, conda

## Licença

Este projeto está sob a licença especificada no arquivo [LICENSE](LICENSE).
