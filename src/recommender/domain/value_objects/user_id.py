from dataclasses import dataclass

@dataclass(frozen=True)
class UserId: 

    """ Identificador unico de um usuário """  

    value: int 

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("UserId must be a positive integer")

