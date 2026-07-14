from dataclasses import dataclass
from datetime import datetime

from recommender.domain.value_objects.hyperparameters import Hyperparameters
from recommender.domain.value_objects.metrics import Metrics


@dataclass(eq=False)
class Experiment:
    """Uma execução de treino/avaliação tem ciclo de vida.

    Identidade = name (estável durante todo o ciclo de vida).
    Igualdade e hash baseados apenas no name"""

    name: str
    model_name: str
    hyperparameters: Hyperparameters
    metrics: Metrics | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Experiment):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


    def start_at(self, started_at: datetime) -> None:
        self.started_at = started_at

    def finish(self, metrics: Metrics, finished_at: datetime) -> None:
        """Transição de estado: o experimento terminou"""
        self.metrics = metrics
        self.finished_at = finished_at

    @property
    def is_finished(self) -> bool:
        return self.metrics is not None