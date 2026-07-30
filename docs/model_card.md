# Model Card — Recommender (MovieLens)

Documento de transparência do modelo, seguindo a prática de *model cards*:
o que o modelo é, para que serve, como foi avaliado e onde falha.

## Detalhes do modelo

- **Projeto:** ecommerce-product-recommender
- **Modelos disponíveis:**
  - `baseline` — recomendador por popularidade (contagem de interações positivas).
  - `pytorch` — Matrix Factorization com embeddings de usuário e item + vieses, treinado com PyTorch.
- **Autor:** Mateus Valentim
- **Data:** 2026-07
- **Licença do código:** MIT. **Dataset:** MovieLens (termos próprios do GroupLens).

## Uso pretendido

- **Objetivo:** dado um usuário, gerar uma lista ordenada de itens (filmes) recomendados.
- **Público-alvo:** demonstração de engenharia de ML (pipeline reprodutível), não produção crítica.
- **Fora de escopo:** recomendação em tempo real de baixa latência; usuários sem histórico (cold start); domínios fora de filmes sem re-treino.

## Dados

- **Dataset:** MovieLens Latest Small (~100 mil avaliações, 610 usuários, ~9,7 mil filmes).
- **Sinais:** avaliações explícitas de 0.5 a 5.0. Interação positiva = nota ≥ 4.0.
- **Split:** temporal **por usuário** (leave-last-out) — as interações mais recentes de cada usuário vão para teste; garante que todo usuário de teste tem histórico no treino.
- **Pré-processamento:** ids esparsos mapeados para índices densos (`IndexEncoder`) para as camadas de embedding.

## Avaliação

Métricas de ranking no corte k=10, média sobre os usuários de teste.

| Métrica | baseline (popularidade) | pytorch (embeddings) |
|---|---|---|
| precision@10 | ~0.031 | ~0.001 |
| recall@10 | ~0.034 | ~0.001 |
| ndcg@10 | ~0.043 | ~0.001 |

> Os valores exatos de cada execução ficam registrados no MLflow.

## Análise crítica

O **baseline de popularidade supera o modelo neural** nas métricas de ranking. Esse resultado é esperado e conhecido na literatura de sistemas de recomendação:

- O modelo neural é treinado com **MSE sobre a nota** (regressão), que otimiza *previsão de rating*, não *ordenação top-k*. Prever bem a nota não implica ranquear bem.
- A **popularidade é um baseline forte** no MovieLens para métricas de ranking.
- Superar a popularidade normalmente exige **treino com perda de ranking** (BPR, WARP) ou **feedback implícito** — o próximo passo natural do projeto.

Valorizamos reportar isso honestamente: o pipeline compara os modelos de forma justa e expõe a limitação, em vez de escondê-la.

## Limitações e riscos

- **Viés de popularidade:** o baseline reforça itens já populares (efeito "rich get richer").
- **Cold start:** o modelo neural não recomenda para usuários/itens ausentes do treino.
- **Esparsidade:** a maioria dos pares usuário-item não tem interação.
- **Objetivo de treino desalinhado com ranking** (ver análise acima).

## Reprodutibilidade

- Seed fixo via `seed_everything`.
- Dependências travadas em `uv.lock`.
- Dados e artefatos versionados com DVC (`dvc repro`).
- Hiperparâmetros e métricas rastreados no MLflow.
