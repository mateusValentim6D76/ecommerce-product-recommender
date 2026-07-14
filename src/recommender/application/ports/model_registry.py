from typing import Protocol, runtime_checkable

from recommender.application.ports.recommendation_model import RecommendationModel


@runtime_checkable
class ModelRegistry(Protocol):
    """Porta: registro versionado de modelos (MLflow Model Registry).

    Diferente do ModelRepository (persistência simples save/get),
    o registry lida com VERSÕES e ESTÁGIOS (Staging/Production) —
    o fluxo de promoção/deploy do modelo.
    """

    def register(self, name: str, model: RecommendationModel) -> str:
        """Registra uma nova versão do modelo. Retorna o id da versão."""
        ...

    def load_production(self, name: str) -> RecommendationModel:
        """Carrega a versão atualmente em produção."""
        ...

    def promote_to_production(self, name: str, version: str) -> None:
        """Promove uma versão para o estágio de produção."""
        ...
