"""Utilitário de early stopping (implementado — pode usar direto).

Early stopping interrompe o treino quando a métrica monitorada para de
melhorar por `patience` épocas seguidas. Evita overfitting e desperdício
de computação. É lógica genérica (não depende de PyTorch), por isso já
vem pronto — você o USA dentro do seu loop de treino.
"""


class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.0) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._best: float | None = None
        self._epochs_without_improvement = 0

    def should_stop(self, current_loss: float) -> bool:
        """Registra a loss da época e diz se o treino deve parar."""
        if self._best is None or current_loss < self._best - self._min_delta:
            self._best = current_loss
            self._epochs_without_improvement = 0
            return False

        self._epochs_without_improvement += 1
        return self._epochs_without_improvement >= self._patience

    @property
    def best_loss(self) -> float | None:
        return self._best
