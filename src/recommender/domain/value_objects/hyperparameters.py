from dataclasses import dataclass, field


@dataclass(frozen=True)
class Hyperparameters:
    """Configuração de entrada de um treino (imutável)

    Flexvel: o conjunto varia por modelo (baseline vsneural)
    """

    values: dict[str, float] = field(default_factory=dict)

    def get(self, name: str) -> float:
        return self.values[name]
