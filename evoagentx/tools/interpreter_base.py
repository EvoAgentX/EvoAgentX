from ..core.module import BaseModule

class BaseInterpreter(BaseModule):
    """
    Base class for interpreter tools that execute code securely.
    Provides common functionality for interpreter operations.
    """

    def __init__(
        self, 
        name: str = 'BaseInterpreter',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

