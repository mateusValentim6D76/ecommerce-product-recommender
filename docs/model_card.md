# Model Card — Recommender

> Preencha ao treinar o modelo final. Model cards documentam o que o
> modelo é, para que serve e onde falha — boa prática de ML responsável.

## Detalhes do modelo
- **Nome / versão:** _(ex.: pytorch-embeddings v1)_
- **Tipo:** _(baseline de popularidade | embeddings PyTorch)_
- **Data do treino:** _(preencher)_
- **Autor:** Mateus Valentim

## Uso pretendido
- **Objetivo:** recomendar itens (filmes MovieLens) a usuários.
- **Usuários-alvo:** _(preencher)_
- **Fora de escopo:** _(ex.: recomendação em tempo real, cold-start severo)_

## Dados
- **Dataset:** MovieLens _(versão: preencher)_
- **Split:** temporal, `test_ratio` padrão 0.2
- **Pré-processamento:** encoding de ids para índices densos; feedback
  explícito (nota) ou implícito (limiar 4.0).

## Métricas
| Métrica | Baseline | Modelo |
|---|---|---|
| precision@10 | _(preencher)_ | _(preencher)_ |
| recall@10 | _(preencher)_ | _(preencher)_ |
| ndcg@10 | _(preencher)_ | _(preencher)_ |

## Limitações e riscos
- _(ex.: viés de popularidade, cold-start, esparsidade)_

## Reprodutibilidade
- Seed fixo (`seed_everything`), `uv.lock` versionado, pipeline `dvc repro`,
  experimentos rastreados no MLflow.
