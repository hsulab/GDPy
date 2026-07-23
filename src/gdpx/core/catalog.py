"""Legacy registry catalog, isolated from bootstrap and core primitives."""


class registers:
    def __init__(self):
        raise RuntimeError("The registers catalog is not intended to be instantiated")

    @staticmethod
    def get(category: str, name: str, convert_name: bool = True):
        if convert_name:
            name = "".join(part.capitalize() for part in name.strip().split("_")) + category.capitalize()
        return getattr(registers, category)[name]

    @staticmethod
    def create(category: str, name: str, convert_name: bool = True, *args, **kwargs):
        return registers.get(category, name, convert_name)(*args, **kwargs)
