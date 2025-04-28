from ..core.module import BaseModule
from ..storages.base import StorageHandler
from .long_term_memory import LongTermMemory


class MemoryManager(BaseModule):
    
    """
    The Memory Manager is responsible for organizing and managing LongTerm Memory's data at a higher level.
    It gets data from LongTermMemory, then it processes the data, store the data in LongTermMemory, 
    and store the LongTermMemory through StorageHandler.
    """

    storage_handler: StorageHandler
    memory: LongTermMemory

