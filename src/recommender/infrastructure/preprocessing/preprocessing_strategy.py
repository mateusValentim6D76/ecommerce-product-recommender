from typing import Protocol, runtime_checkable

from recommender.domain.value_objects.interaction import Interaction


@runtime_checkable
class FeedbackStrategy(Protocol):
    """Strategy: converte uma interação no 'alvo' de treino.

    Permite tratar o MovieLens como feedback EXPLÍCITO (usar a nota)
    ou IMPLÍCITO (gostei/não gostei), sem espalhar if/else pelo código.
    """

    def target(self, interaction: Interaction) -> float:
        ...


class ExplicitFeedbackStrategy:
    """Alvo = a própria nota (0.5–5.0). Problema de regressão."""

    def target(self, interaction: Interaction) -> float:
        return interaction.rating.value


class ImplicitFeedbackStrategy:
    """Alvo = 1.0 se a interação é positiva, senão 0.0. Classificação."""

    def __init__(self, threshold: float = 4.0) -> None:
        self._threshold = threshold

    def target(self, interaction: Interaction) -> float:
        return 1.0 if interaction.is_positive(self._threshold) else 0.0
