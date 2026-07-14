"""[VOCÊ IMPLEMENTA] Baseline de popularidade.

Este é o modelo mais simples possível: recomenda para TODO mundo os
itens mais populares (mais avaliados / com melhor média). Ele existe
para ser a RÉGUA: se o modelo PyTorch não superar este baseline, ele
não está agregando valor. Baseline é obrigatório em ML Engineering.

Você preenche os dois métodos marcados com NotImplementedError.
"""

from collections import Counter  # dica: ótimo para contar popularidade

from recommender.domain.entities.recommendation import Recommendation, RecommendedItem
from recommender.domain.value_objects.hyperparameters import Hyperparameters
from recommender.domain.value_objects.interaction import Interaction
from recommender.domain.value_objects.item_id import ItemId
from recommender.domain.value_objects.score import Score
from recommender.domain.value_objects.user_id import UserId


class PopularityModel:
    """Satisfaz a porta RecommendationModel.

    Guarda um ranking global de itens (do mais para o menos popular) e
    devolve os top-k para qualquer usuário.
    """

    def __init__(self, ranked_items: list[RecommendedItem]) -> None:
        self._ranked_items = ranked_items

    def recommend(self, user_id: UserId, k: int) -> Recommendation:
        # DICA: como o ranking é global, a recomendação é a mesma para
        # todos os usuários. Basta devolver os k primeiros.
        # Sugestão: return Recommendation(user_id, self._ranked_items[:k])
        raise NotImplementedError("Implemente a recomendação por popularidade")


class BaselineTrainer:
    """Satisfaz a porta ModelTrainer.

    'Treinar' aqui é só contar popularidade — não há gradiente.
    """

    def train(
        self,
        interactions: list[Interaction],
        hyperparameters: Hyperparameters,
    ) -> PopularityModel:
        # PASSO A PASSO sugerido:
        # 1. Conte quantas interações (positivas?) cada ItemId recebeu.
        #    Ex.: counts = Counter(i.item_id for i in interactions)
        # 2. Ordene os itens por contagem (desc).
        # 3. Normalize a contagem para virar um Score (>= 0). Ex.: dividir
        #    pela maior contagem para ficar em [0, 1].
        # 4. Monte uma list[RecommendedItem] e devolva PopularityModel(...).
        raise NotImplementedError("Implemente o treino do baseline de popularidade")
