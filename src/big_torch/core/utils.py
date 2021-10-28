class ModuleAggregator:
    def __init__(self) -> None:
        self._registry = {}

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._registry[key]
        return key

    def register(self, name=None):
        def wrapper(func):
            self._registry[name if name != None else func.__name__] = func
            return func

        return wrapper
