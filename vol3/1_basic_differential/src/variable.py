from typing import Any

class Variable:
    def __init__(self, data: Any) -> None:
        self.data: Any = data

    def get_data(self) -> Any:
        return self.data

    def set_data(self, new_data: Any) -> None:
        self.data = new_data
