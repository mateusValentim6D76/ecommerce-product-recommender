from typing import Protocol, runtime_checkable

from recommender.domain.entities.experiment import Experiment


@runtime_checkable
class ExperimentTracker(Protocol):
    """Porta: contrato para rastrear experimentos (MLflow & cia).

    O núcleo entrega um Experiment finalizado (com hiperparâmetros
    e métricas) e não sabe se por trás há MLflow, W&B ou um arquivo.
    """

    def track(self, experiment: Experiment) -> None:
        """Registra hiperparâmetros, métricas e metadados do experimento."""
        ...
