from recommender.domain.entities.recommendation import Recommendation
from recommender.domain.repositories.interaction_repository import InteractionRepository
from recommender.domain.repositories.model_repository import ModelRepository
from recommender.domain.services.recommendation_service import RecommendationService
from recommender.domain.value_objects.user_id import UserId


class GenerateRecommendation:
    """Carrega o modelo, gera recomendações e opcionalmente filtra itens vistos."""

    def __init__(
        self,
        model_repository: ModelRepository,
        interaction_repository: InteractionRepository,
        recommendation_service: RecommendationService | None = None,
    ) -> None:
        self._model_repository = model_repository
        self._interaction_repository = interaction_repository
        self._recommendation_service = recommendation_service or RecommendationService()

    def execute(
        self,
        model_name: str,
        user_id: UserId,
        k: int = 10,
        filter_seen: bool = True,
    ) -> Recommendation:
        model = self._model_repository.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")

        # Pede mais candidatos quando vai filtrar, para ainda sobrar k.
        candidates = model.recommend(user_id, k * 2 if filter_seen else k)

        if filter_seen:
            history = self._interaction_repository.for_user(user_id)
            candidates = self._recommendation_service.filter_already_seen(
                candidates, history
            )

        return candidates.top_k(k)
