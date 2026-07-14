from recommender.application.ports.model_trainer import ModelTrainer
from recommender.application.ports.recommendation_model import RecommendationModel
from recommender.domain.repositories.model_repository import ModelRepository
from recommender.domain.value_objects.hyperparameters import Hyperparameters
from recommender.domain.value_objects.interaction import Interaction


class TrainModel:
    """Caso de uso: treina um modelo e o persiste.

    Depende das PORTAS ModelTrainer (como treinar) e ModelRepository
    (onde guardar). Não sabe se é baseline ou PyTorch, nem se o
    repositório é memória, disco ou MLflow.
    """

    def __init__(self, trainer: ModelTrainer, model_repository: ModelRepository) -> None:
        self._trainer = trainer
        self._model_repository = model_repository

    def execute(
        self,
        model_name: str,
        interactions: list[Interaction],
        hyperparameters: Hyperparameters,
    ) -> RecommendationModel:
        model = self._trainer.train(interactions, hyperparameters)
        self._model_repository.save(model_name, model)
        return model
