from dataclasses import dataclass

@dataclass(frozen=True)
class Rating:

    """ Nota que um usuario atribui a um item.
    
    Faixa valida do movien lens 0.5 até 5.0
    Imutavel e validado na contrução
    """

    value : float

    
    def __post_init__(self) -> None:
        if not 0.5 <= self.value <= 5.0:
            raise ValueError("Rating must be between 0.5 and 5.0")
        
        
    def is_positive(self, threshold: float = 4.0) -> bool:
        """Indica se a nota representa uma interação positiva"""
        return self.value >= threshold
    