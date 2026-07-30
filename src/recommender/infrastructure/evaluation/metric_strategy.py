"""métricas de ranking

Cada função recebe:
- recommended: lista de ItemId recomendados, JÁ ordenados (melhor primeiro);
- relevant: conjunto de ItemId realmente relevantes para o usuário (no teste);
- k: corte.
E devolve um float. São funções puras (fáceis de testar isoladamente).

Definições rápidas:
- precision@k = (# recomendados relevantes nos top-k) / k
- recall@k    = (# recomendados relevantes nos top-k) / (# relevantes)
- ndcg@k      = DCG / IDCG, onde DCG soma 1/log2(pos+1) para cada acerto
"""

import math

from recommender.domain.value_objects.item_id import ItemId


def precision_at_k(recommended: list[ItemId], relevant: set[ItemId], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list[ItemId], relevant: set[ItemId], k: int) -> float:
    # acertos nos top-k dividido pelo total de relevantes
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[ItemId], relevant: set[ItemId], k: int) -> float:
    # DCG = soma de 1/log2(i+2) para cada posição i (0-indexed) qual
    # item esteja em `relevant`. IDCG = DCG do ranking ideal. ndcg = DCG/IDCG.
    # (import math; math.log2)
    if not relevant:
        return 0.0
    dcg = 0.0
    for position, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / math.log2(position + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
