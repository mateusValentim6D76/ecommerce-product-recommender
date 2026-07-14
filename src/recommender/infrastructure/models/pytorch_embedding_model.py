"""[VOCÊ IMPLEMENTA] Modelo neural de embeddings (PyTorch).

Ideia central de um recomendador por embeddings:
- cada usuário vira um vetor (embedding) aprendido;
- cada item vira um vetor (embedding) aprendido;
- a afinidade usuário-item é o produto escalar dos dois vetores.
O treino ajusta os vetores para que o produto escalar se aproxime do
alvo (nota, no feedback explícito).

Você preenche:
1. A rede `EmbeddingRecommender` (nn.Module): __init__ e forward.
2. O adapter `TorchRecommendationModel`: como transformar a rede
   treinada + os encoders em uma Recommendation do domínio.
"""

import torch
import torch.nn as nn

from recommender.domain.entities.recommendation import Recommendation, RecommendedItem
from recommender.domain.value_objects.item_id import ItemId
from recommender.domain.value_objects.score import Score
from recommender.domain.value_objects.user_id import UserId
from recommender.infrastructure.preprocessing.movielens_preprocessor import IndexEncoder


class EmbeddingRecommender(nn.Module):
    """A rede neural em si."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32) -> None:
        super().__init__()
        # TODO: defina as camadas. Dica:
        # self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # self.item_embedding = nn.Embedding(num_items, embedding_dim)
        # (opcional) self.user_bias / self.item_bias = nn.Embedding(n, 1)
        raise NotImplementedError("Defina as camadas de embedding")

    def forward(self, user_index: torch.Tensor, item_index: torch.Tensor) -> torch.Tensor:
        # TODO: retorne o score previsto. Dica:
        # u = self.user_embedding(user_index)
        # v = self.item_embedding(item_index)
        # return (u * v).sum(dim=1)   # produto escalar
        raise NotImplementedError("Implemente o forward (produto escalar)")


class TorchRecommendationModel:
    """Satisfaz a porta RecommendationModel usando a rede treinada.

    Faz a ponte entre o mundo de tensores (índices) e o domínio (ids).
    """

    def __init__(
        self,
        network: EmbeddingRecommender,
        user_encoder: IndexEncoder[UserId],
        item_encoder: IndexEncoder[ItemId],
    ) -> None:
        self._network = network
        self._user_encoder = user_encoder
        self._item_encoder = item_encoder

    def recommend(self, user_id: UserId, k: int) -> Recommendation:
        # PASSO A PASSO sugerido:
        # 1. self._network.eval(); with torch.no_grad():
        # 2. Pegue o índice do usuário: self._user_encoder.index_of(user_id)
        #    (trate usuário desconhecido -> pode cair no baseline ou lista vazia)
        # 3. Monte tensores para TODOS os itens e calcule os scores de uma vez.
        # 4. Ordene, pegue top-k, traduza índice->ItemId com self._item_encoder.
        # 5. Monte RecommendedItem(ItemId, Score(valor)) e devolva Recommendation.
        raise NotImplementedError("Implemente a geração de recomendações")
