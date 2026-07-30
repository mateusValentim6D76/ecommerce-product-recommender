# ecommerce-product-recommender

End-to-end recommendation system using MovieLens data, PyTorch, DVC, MLflow, Docker and clean code practices.

Sistema de recomendação com **arquitetura hexagonal + DDD leve**: o domínio é puro
(sem pandas/torch/mlflow) e a infraestrutura se conecta a ele através de **portas**
(interfaces). Inclui um baseline de popularidade e um modelo neural de embeddings
(PyTorch), comparados de forma justa.

## Arquitetura

```
src/recommender/
├── domain/            # núcleo puro: entities, value objects, repositories (contratos), services
├── application/       # casos de uso, DTOs e portas (contratos que a infra implementa)
├── infrastructure/    # adapters: pandas (dados), torch (modelo), mlflow (tracking/registry)
├── interfaces/        # API (FastAPI) e CLI (composição / entrypoints)
└── shared/            # utilidades (seed, exceptions)
```

Regra de dependência: tudo aponta **para dentro**. `infrastructure` e `interfaces`
dependem de `application`/`domain`; o domínio não depende de ninguém. Detalhes em
[docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md).

## Pré-requisitos

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Docker + Docker Compose (opcional, para a stack containerizada)

## Dataset

O MovieLens **não** é redistribuído no repositório (tem termos próprios) e é
versionado com DVC. Obtenha os dados de uma destas formas:

- **Com DVC remote configurado:** `dvc pull`
- **Manualmente:** baixe o MovieLens Latest Small
  (https://grouplens.org/datasets/movielens/) e coloque em:
  ```
  data/raw/movies.csv
  data/raw/ratings.csv
  ```

---

## Execução local

### 1. Instalar dependências
```bash
uv sync
uv run python scripts/validate_env.py
```

### 2. Treinar e avaliar
```bash
# valida a leitura dos dados e mostra o tamanho do split
uv run python -m recommender.interfaces.cli.preprocess --data-dir data/raw

# treina + avalia + registra no MLflow (sqlite:///mlflow.db)
uv run python -m recommender.interfaces.cli.train --model-type baseline --model-name baseline-v1
uv run python -m recommender.interfaces.cli.train --model-type pytorch  --model-name pytorch-v1

# gera recomendações para um usuário
uv run python -m recommender.interfaces.cli.recommend --model-name baseline-v1 --user-id 1 --k 10
```
Os modelos treinados são salvos em `models/<nome>.pkl`.

### 3. Comparar experimentos no MLflow
```bash
uv run mlflow ui
```
Abra `http://localhost:5000` e selecione o experimento **`recommender`**.

### 4. Subir a API de inferência
A API carrega um modelo já treinado de `models/`. Informe qual via `MODEL_NAME`:
```bash
# Linux/macOS
MODEL_NAME=baseline-v1 uv run uvicorn recommender.interfaces.api.app:app --reload
```
```powershell
# Windows PowerShell
$env:MODEL_NAME="baseline-v1"; uv run uvicorn recommender.interfaces.api.app:app --reload
```
Endpoints:
- `GET /health`
- `GET /users/{user_id}/recommendations?k=10`
- `GET /docs` documentação interativa (Swagger UI)

---

## Execução com Docker

Sobe a stack completa: **API** (porta 8000) + servidor **MLflow** (porta 5000).

> Pré-requisito: ter os modelos em `./models` (rode o treino local antes, ou
> `dvc pull`). O `compose` monta `./data` e `./models` no container e serve o
> modelo definido em `MODEL_NAME` (padrão `baseline-v1`).

```bash
docker compose up --build
```
Depois:
```bash
curl http://localhost:8000/health
curl "http://localhost:8000/users/1/recommendations?k=5"
```
- API: `http://localhost:8000` (docs em `/docs`)
- MLflow: `http://localhost:5000`

Para parar: `docker compose down`.

---

## Pipeline reprodutível (DVC)

Dados (`data/raw/*.csv`) e modelos (`models/*.pkl`) são versionados com DVC.
```bash
dvc repro          # executa treino + avaliação a partir dos dados versionados
dvc add models/baseline-v1.pkl   # versiona um artefato de modelo
dvc push           # envia dados/modelos para o remote (se configurado)
```

## Testes e qualidade

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

## Resultados

O baseline de popularidade supera o modelo neural (MF + MSE) nas métricas de
ranking: resultado esperado, discutido em [docs/model_card.md](docs/model_card.md).

## Licença

MIT (código). O dataset MovieLens possui termos próprios ver a fonte oficial.
