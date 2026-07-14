from typing import Protocol, runtime_checkable

from recommender.application.ports.recommendation_model import RecommendationModel


@runtime_checkable
class ModelRepository(Protocol):
    """Repositório: persistência simples de modelos treinados.

    save/get por nome. É a abstração que os casos de uso usam para
    guardar o modelo após o treino e recarreg-lo na inferência.
    (Para versionamento/estágios, veja a porta ModelRegistry.)
    """

    def save(self, name: str, model: RecommendationModel) -> None:
        """Persiste um modelo treinado sob um nome."""
        ...

    def get(self, name: str) -> RecommendationModel | None:
        """Recupera um modelo pelo nome. None se não existir."""
        ...
