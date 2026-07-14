# Imagem da API de inferência.
FROM python:3.13-slim

# uv (gerenciador de pacotes) copiado da imagem oficial.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Instala dependências primeiro (melhor cache de camadas).
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv sync --no-dev

EXPOSE 8000

# Sobe a API FastAPI via uvicorn.
CMD ["uv", "run", "--no-dev", "uvicorn", \
     "recommender.interfaces.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
