import importlib

from .base import DBStoreBase
from ..storages_config import DBConfig

__all__ = ['DBStoreBase', 'SQLite', 'DBStoreFactory']

def load_class(class_type: str):
    """
    Dynamically load a class from a module path.

    Attributes:
        class_type (str): Fully qualified class path (e.g., 'module.submodule.ClassName').

    Returns:
        type: The loaded class.

    Raises:
        ImportError: If the module or class cannot be imported.
        AttributeError: If the class is not found in the module.
    """
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class DBStoreFactory:
    """
    Factory class for creating database store instances based on provider and configuration.
    Maps provider names to specific database store classes.
    """
    provider_to_class = {
        "sqlite": "evoagentx.storages.db_stores.sqlite.SQLite",
        "posgre_sql": "evoagentx.storages.db_stores.posgre_sql.",  # Note: Incomplete path, likely a placeholder
    }

    @classmethod
    def create(cls, provider_name: str, config: DBConfig):
        """
        Create a database store instance for the specified provider.

        Attributes:
            provider_name (str): Name of the database provider (e.g., 'sqlite', 'posgre_sql').
            config (DBConfig): Configuration for the database store.

        Returns:
            DBStoreBase: An instance of the database store.

        Raises:
            ValueError: If the provider is not supported.
        """
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            if not isinstance(config, dict):
                config = config.model_dump()
            db_store_class = load_class(class_type)
            return db_store_class(**config)
        else:
            raise ValueError(f"Unsupported Database provider: {provider_name}")