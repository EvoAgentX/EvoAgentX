from ..core.module import BaseModule
from ..storages.base import StorageHandler
from .long_term_memory import LongTermMemory


class MemoryManager(BaseModule):
    
    """
    The Memory Manager is responsible for organizing and managing LongTerm Memory's data at a higher level.
    It gets data from LongTermMemory, then it processes the data, store the data in LongTermMemory, 
    and store the LongTermMemory through StorageHandler.

    Function examples:
    - Load data from LongTermMemory and convert records into different memory units (e.g., chunk, KG, etc) 
    - Store the memory units in LongTermMemory
    - Store LongTermMemory 
    - Build indexes for data in LongTermMemory based on different strategies or structures (such as keyword indexes, vector indexes)
    - Fast retrieval based on index
    """

    storage_handler: StorageHandler
    memory: LongTermMemory

