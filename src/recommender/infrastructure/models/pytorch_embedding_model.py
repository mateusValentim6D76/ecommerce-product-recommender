"""Modelo neural de embeddings (Matrix Factorization com vieses).

A nota prevista é o produto escalar dos embeddings de usuário e item,
somado aos vieses de usuário, item e global.
"""

import torch
from torch import nn

from recommender.domain.entities.recommendation import Recommendation, RecommendedItem
from recommender.domain.value_objects.item_id import ItemId
from recommender.domain.value_objects.score import Score
from recommender.domain.value_objects.user_id import UserId
from recommender.infrastructure.preprocessing.movielens_preprocessor import IndexEncoder


class EmbeddingRecommender(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_index: torch.Tensor, item_index: torch.Tensor) -> torch.Tensor:
        user_vector = self.user_embedding(user_index)
        item_vector = self.item_embedding(item_index)
        dot = (user_vector * item_vector).sum(dim=1)
        return (
            dot
            + self.user_bias(user_index).squeeze(1)
            + self.item_bias(item_index).squeeze(1)
            + self.global_bias
        )


class TorchRecommendationModel:
    """Implementa RecommendationModel usando a rede treinada e os encoders."""

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
        if user_id not in self._user_encoder:
            return Recommendation(user_id=user_id, items=[])

        self._network.eval()
        with torch.no_grad():
            num_items = len(self._item_encoder)
            user_index = torch.full((num_items,), self._user_encoder.index_of(user_id))
            item_index = torch.arange(num_items)

            scores = self._network(user_index, item_index)
            scores = torch.clamp(scores, min=0.0)

            top_values, top_indices = torch.topk(scores, min(k, num_items))

        items = [
            RecommendedItem(
                item_id=self._item_encoder.id_of(int(index)),
                score=Score(float(value)),
            )
            for value, index in zip(top_values, top_indices, strict=False)
        ]
        return Recommendation(user_id=user_id, items=items)
