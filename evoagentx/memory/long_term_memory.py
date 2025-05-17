
from .memory import BaseMemory
from ..storages.base import StorageHandler


class LongTermMemory(BaseMemory):

    """
    Responsible for the management of raw data for long-term storage.
    """
    storage: StorageHandler
    # rag_engine = ...
    pass 


