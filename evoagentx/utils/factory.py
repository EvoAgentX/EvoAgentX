import importlib

from ..storages.storages_config import DBConfig, VectorStoreConfig


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


class VectorStoreFactory:
    """
    Factory class for creating vector store instances based on configuration.
    Maps provider names to specific vector store classes.
    """
    provider_to_class = {
        "qdrant": "mem0.vector_stores.qdrant.Qdrant",
        "chroma": "mem0.vector_stores.chroma.ChromaDB",
        "faiss": "mem0.vector_stores.faiss.FAISS",
    }


    def create(cls, config: VectorStoreConfig):
        """
        Create a vector store instance based on the provided configuration.

        Attributes:
            config (VectorStoreConfig): Configuration for the vector store.

        Returns:
            VectorStoreBase: An instance of the vector store.
        """
        # TODO: Implement vector store creation logic
        pass


# Factory for creating graph store instances
class GraphStoreFactory:
    """
    Factory class for creating graph store instances based on configuration.
    Maps provider names to specific graph store classes.
    """
    provider_to_class = {
        "neo4j": "mem0.graph_stores",  # Note: Incomplete mapping, likely a placeholder
        "": "mem0.graph_stores",  # Note: Incomplete mapping, likely a placeholder
    }

    @classmethod
    def create(cls, config: VectorStoreConfig):
        """
        Create a graph store instance based on the provided configuration.

        Attributes:
            config (VectorStoreConfig): Configuration for the graph store.

        Returns:
            GraphStoreBase: An instance of the graph store.
        """
        # TODO: Implement graph store creation logic
        pass