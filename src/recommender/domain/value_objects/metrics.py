from dataclasses import dataclass, field


@dataclass(frozen=True)
class Metrics:
    """Resultado de avaliação de um experimento (imutável)."""

    values: dict[str, float] = field(default_factory=dict)

    def get(self, name: str) -> float:
        return self.values[name]
