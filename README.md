# ecommerce-product-recommender

End-to-end recommendation system using MovieLens data, PyTorch, DVC, MLflow, Docker and clean code practices.

Sistema de recomendação construído com **arquitetura hexagonal + DDD leve**: o
domínio é puro (sem pandas/torch/mlflow), e a infraestrutura se pluga nele através
de **portas** (interfaces).

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
dependem de `application`/`domain`; o domínio não depende de ninguém.

## Setup

```bash
uv sync                      # cria o .venv e instala tudo (inclui torch, mlflow, fastapi)
uv run python scripts/validate_env.py
```

### Dataset

O MovieLens **não** é redistribuído aqui (tem termos próprios). Baixe da fonte
oficial (https://grouplens.org/datasets/movielens/) e coloque em:

```
data/raw/movies.csv
data/raw/ratings.csv
```

## Uso

```bash
# Pré-processar (valida dados + mostra tamanho do split)
uv run python -m recommender.interfaces.cli.preprocess --data-dir data/raw

# Treinar (baseline ou pytorch) + avaliar + registrar no MLflow
uv run python -m recommender.interfaces.cli.train --model-type baseline --model-name baseline-v1

# Recomendar para um usuário
uv run python -m recommender.interfaces.cli.recommend --model-name baseline-v1 --user-id 1

# Subir a API de inferência
uv run uvicorn recommender.interfaces.api.app:app --reload
#  GET /health
#  GET /users/{user_id}/recommendations?k=10

# Pipeline reprodutível (DVC) e stack completa (Docker)
dvc repro
docker compose up --build
```

## Testes

```bash
uv run pytest
```

## Qualidade

```bash
uv run ruff check .
uv run ruff format .
```

## Licença

MIT (código). O dataset MovieLens possui termos próprios — ver fonte oficial.
