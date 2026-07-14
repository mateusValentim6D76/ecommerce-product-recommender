from typing import Protocol, runtime_checkable

from recommender.domain.entities.experiment import Experiment


@runtime_checkable
class ExperimentRepository(Protocol):
    """Repositório para persistir e recuperar experimentos.

    Diferente do InteractionRepository (só leitura), aqui a
    aplicação PRODUZ experimentos e precisa guardar los.
    """

    def save(self, experiment: Experiment) -> None:
        """Persiste (ou atualiza) um experimento."""
        ...

    def get_by_name(self, name: str) -> Experiment | None:
        """Recupera pelo nome (identidade). None se não existir."""
        ...

    def all(self) -> list[Experiment]:
        """Todos os experimentos util para comparar modelos."""
        ...
