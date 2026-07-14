from dataclasses import dataclass

@dataclass(frozen=True)
class ItemId:
    """Identificador unico de um item de catalogo.

        Value object imutavel: validado na construção, de modo que
        qualquer ItemId existente é por definição válido.
        """

    value: int

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("ItemId deve ser um inteiro positov" )
