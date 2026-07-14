from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrainModelInput:
    """Entrada do caso de uso de treino.

    DTO (Data Transfer Object): carrega dados 'crus' entre as bordas
    da aplicação (CLI/API -> use case). NÃO é um objeto de domínio —
    por isso usa tipos primitivos (str, dict), não value objects.
    """

    model_type: str  # "baseline" | "pytorch"
    model_name: str
    hyperparameters: dict[str, float] = field(default_factory=dict)
    test_ratio: float = 0.2
    k: int = 10
