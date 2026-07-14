from recommender.domain.entities.experiment import Experiment


class InMemoryExperimentRepository:
    """Implementação simples de ExperimentRepository (em memória).

    A chave é o nome do experimento (sua identidade). save é um
    upsert: salvar com nome existente substitui.
    """

    def __init__(self) -> None:
        self._by_name: dict[str, Experiment] = {}

    def save(self, experiment: Experiment) -> None:
        self._by_name[experiment.name] = experiment

    def get_by_name(self, name: str) -> Experiment | None:
        return self._by_name.get(name)

    def all(self) -> list[Experiment]:
        return list(self._by_name.values())
