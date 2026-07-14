from dataclasses import dataclass

@dataclass(frozen=True)
class Score:

    """Relevancia prevista por um modelo para um item.

    Diferente de Rating (entrada humana), Score é saída do modelo.
    Não tem teto fixo depende do modelo mas nunca é negativo.
    Não se auto-ordena: ranquear é responsabilidade da Recommendation"""

    value: float

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("Score must be grather than or equal to zero")
        
    


