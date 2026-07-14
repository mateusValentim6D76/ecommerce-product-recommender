from recommender.application.ports.recommendation_model import RecommendationModel


class InMemoryModelRepository:
    """Implementação simples de ModelRepository (dicionário em memória).

    Útil para testes e execução local sem MLflow. Troca-se por uma
    implementação de disco/MLflow sem o núcleo perceber.
    """

    def __init__(self) -> None:
        self._models: dict[str, RecommendationModel] = {}

    def save(self, name: str, model: RecommendationModel) -> None:
        self._models[name] = model

    def get(self, name: str) -> RecommendationModel | None:
        return self._models.get(name)
