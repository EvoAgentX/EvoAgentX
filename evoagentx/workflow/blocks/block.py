
class block:
    def __init__(self) -> None:
        self.n = 0
    
    def execute(self, problem):

        raise NotImplementedError

    async def async_execute(self, problem):

        raise NotImplementedError

    def save(self, path: str):

        raise NotImplementedError
    
    def load(self, path: str):

        raise NotImplementedError