import json
from typing import Any, List, Sequence, Union

from llama_index.core.schema import BaseNode, TextNode
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.graph_stores.types import (
    PropertyGraphStore,
    LabelledNode,
    Relation,
    EntityNode,
    ChunkNode,
)

from .base import GraphStoreBase
from evoagentx.core.logging import logger

# For fixing the bugs in adding entity.
CHUNK_SIZE = 1000
BASE_ENTITY_LABEL = "__Entity__"
BASE_NODE_LABEL = "__Node__"
class BasicNeo4jStore(Neo4jPropertyGraphStore):
    def __init__(self, username: str, password: str, url: str, database: str | None = "neo4j", refresh_schema: bool = True, sanitize_query_output: bool = True, enhanced_schema: bool = False, create_indexes: bool = True, timeout: float | None = None, **neo4j_kwargs: Any) -> None:
        super().__init__(username, password, url, database, refresh_schema, sanitize_query_output, enhanced_schema, create_indexes, timeout, **neo4j_kwargs)

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        # Lists to hold separated types
        entity_dicts: List[dict] = []
        chunk_dicts: List[dict] = []

        # Sort by type
        for item in nodes:
            if isinstance(item, EntityNode):
                entity_dicts.append({**item.model_dump(), "id": item.id})
            elif isinstance(item, ChunkNode):
                chunk_dicts.append({**item.model_dump(), "id": item.id})
            else:
                # Log that we do not support these types of nodes
                # Or raise an error?
                pass

        if chunk_dicts:
            for index in range(0, len(chunk_dicts), CHUNK_SIZE):
                chunked_params = chunk_dicts[index : index + CHUNK_SIZE]
                self.structured_query(
                    f"""
                    UNWIND $data AS row
                    MERGE (c:{BASE_NODE_LABEL} {{id: row.id}})
                    SET c.text = row.text, c:Chunk
                    WITH c, row
                    SET c += row.properties
                    WITH c, row.embedding AS embedding
                    WHERE embedding IS NOT NULL
                    CALL db.create.setNodeVectorProperty(c, 'embedding', embedding)
                    RETURN count(*)
                    """,
                    param_map={"data": chunked_params},
                )

        if entity_dicts:
            for index in range(0, len(entity_dicts), CHUNK_SIZE):
                chunked_params = entity_dicts[index : index + CHUNK_SIZE]
                self.structured_query(
                    f"""
                    UNWIND $data AS row
                    MERGE (e:{BASE_NODE_LABEL} {{id: row.id}})
                    SET e += apoc.map.clean(row.properties, [], [])
                    SET e.name = row.name, e:`{BASE_ENTITY_LABEL}`
                    WITH e, row
                    CALL apoc.create.addLabels(e, [row.label])
                    YIELD node
                    WITH e, row
                    CALL (e, row) {{
                        WITH e, row
                        WHERE row.embedding IS NOT NULL
                        CALL db.create.setNodeVectorProperty(e, 'embedding', row.embedding)
                        RETURN count(*) AS count
                    }}
                    WITH e, row WHERE row.properties.triplet_source_id IS NOT NULL
                    MERGE (c:{BASE_NODE_LABEL} {{id: row.properties.triplet_source_id}})
                    MERGE (e)<-[:MENTIONS]-(c)
                    """,
                    param_map={"data": chunked_params},
                )

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Add relations."""
        params = [r.model_dump() for r in relations]
        for index in range(0, len(params), CHUNK_SIZE):
            chunked_params = params[index : index + CHUNK_SIZE]

            self.structured_query(
                f"""
                UNWIND $data AS row
                MERGE (source: {BASE_NODE_LABEL} {{id: row.source_id}})
                ON CREATE SET source:Chunk
                MERGE (target: {BASE_NODE_LABEL} {{id: row.target_id}})
                ON CREATE SET target:Chunk
                WITH source, target, row
                CALL apoc.merge.relationship(source, row.label, {{}}, row.properties, target) YIELD rel
                RETURN count(*)
                """,
                param_map={"data": chunked_params},
            )


class Neo4jGraphStoreWrapper(GraphStoreBase):
    """Wrapper for Neo4j graph store."""
    
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        **kwargs
    ):
        try:
            self.graph_store = BasicNeo4jStore(
                url=uri,
                username=username,
                password=password,
                database=database,
            )
        except Exception as e:
            raise ValueError(f"Failed to connect to Neo4j: {str(e)}")

        self.verify_version()
    
    def get_graph_store(self) -> PropertyGraphStore:
        return self.graph_store
    
    @property
    def supports_vector_queries(self):
        return self.graph_store.supports_vector_queries and \
            self.graph_store._supports_vector_index

    # borrow from llama_index
    def verify_version(self):
        """
        Check if the connected Neo4j database version supports vector indexing
        without specifying embedding dimension.

        Queries the Neo4j database to retrieve its version and compares it
        against a target version (5.23.0) that is known to support vector
        indexing. 
        """
        db_data = self.graph_store.structured_query("CALL dbms.components()")
        version = db_data[0]["versions"][0]
        if "aura" in version:
            version_tuple = (*map(int, version.split("-")[0].split(".")), 0)
        else:
            version_tuple = tuple(map(int, version.split(".")))

        target_version = (5, 23, 0)

        if version_tuple >= target_version:
            self.graph_store._supports_vector_index = True
        else:
            self.graph_store._supports_vector_index = False
            logger.warning(f"The version of Neo4j server is {version_tuple}, which is less than {target_version}. Disable the vector indexing.")

    def clear(self) -> None:
        """
        Clear the node and relation in the neo4j graph database.
        """
        with self.graph_store.client.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            session.run("CALL apoc.schema.assert({}, {})")

    async def aload(self, node: Union[LabelledNode, Relation, BaseNode]) -> None:
        """
        Asynchronously load a single node into the Neo4j graph database.

        Checks if a node with the same ID already exists in the database. If it does not exist,
        inserts the node as either an EntityNode or ChunkNode based on its type. Handles metadata
        and embeddings appropriately.

        Args:
            node (Union[LabelledNode, Relation, BaseNode]): The node/relation to load, either a Chunk or a LlamaIndex BaseNode.
        """
        try:

            if not isinstance(node, (BaseNode, EntityNode, ChunkNode, Relation)):
                raise ValueError(f"Unsupported node type: {type(node)}. Must be BaseNode, EntityNode, ChunkNode, Relation.")
            
            if isinstance(node, (EntityNode, ChunkNode)):
                self.graph_store.upsert_nodes([node])
            elif isinstance(node, BaseNode):
                self.graph_store.upsert_llama_nodes([node])
            elif isinstance(node, Relation):
                self.graph_store.upsert_relations([node])

            if self.graph_store.supports_structured_queries:
                self.graph_store.get_schema(refresh=True)  

        except Exception as e:
            logger.error(f"Failed to load node with ID {node.id} into Neo4j: {str(e)}")
            raise

    def build_kv_store(self) -> Sequence[Union[LabelledNode, EntityNode, ChunkNode, Relation]]:
        """
        Build a kv_store from neo4j database.
        Returns a dictionary where:
        - Key: node ID
        - Value: Node object (EntityNode, ChunkNode, Relation)
        """
        try:
            # For embedding could query successfully
            cur_sanitize_query_output = self.graph_store.sanitize_query_output
            self.graph_store.sanitize_query_output = False
            
            # Query all nodes with their labels, properties, and embeddings
            nodes_query = f"""
                MATCH (n:{BASE_NODE_LABEL})
                RETURN n.id AS name, labels(n) AS labels,
                       n.text AS text,
                       n.embedding AS embedding,
                       properties(n) AS properties
            """
            nodes_result = self.graph_store.structured_query(nodes_query)

            nodes = []
            for record in nodes_result:
                labels = record["labels"]
                node_dict = {
                    "id": record["name"],
                    "labels": labels,
                    "embedding": record["embedding"],
                    "properties": record["properties"]
                }
                if "Chunk" in labels:
                    # Handle ChunkNode attributes
                    # Use the BaseNode to handle this
                    if node_dict['properties']['_node_type'] == 'TextNode':
                        content = json.loads(node_dict['properties']['_node_content'])
                        content['metadata'] = json.loads(content['metadata']['metadata'])
                        node = TextNode(**content)

                    nodes.append(node)

                elif BASE_ENTITY_LABEL in labels:
                    # Handle EntityNode attributes
                    node_dict["name"] = record["name"] or record["id"]
                    node_dict["label"] = [label for label in labels if label not in [BASE_NODE_LABEL, BASE_ENTITY_LABEL]][0] if any(label not in [BASE_NODE_LABEL, BASE_ENTITY_LABEL] for label in labels) else "entity"
                    node = EntityNode(
                        name=node_dict["name"],
                        label=node_dict["label"],
                        embedding=node_dict["embedding"],
                        properties={"triplet_source_id": node_dict["properties"]["triplet_source_id"]}
                    )
                    nodes.append(node)
                else:
                    logger.warning(f"Skipping node with id {record['id']} due to unsupported labels: {labels}")
                    continue

            # Query all relations
            relations_query = """MATCH ()-[r]->() RETURN type(r) AS label, startNode(r).id AS source_id, endNode(r).id AS target_id, properties(r) AS properties"""
            relations_result = self.graph_store.structured_query(relations_query)
            relations = [
                Relation(
                    label=record["label"],
                    source_id=record["source_id"],
                    target_id=record["target_id"],
                    properties=json.loads(record["properties"].get("metadata", {})) if isinstance(record["properties"].get("metadata", {}), str) else record["properties"].get("metadata", {}),
                )
                for record in relations_result
            ]

            # Reset as the same as before
            self.graph_store.sanitize_query_output = cur_sanitize_query_output

            logger.info(f"Exported {len(nodes)} nodes and {len(relations)} relations from Neo4j graph store")
            return nodes + relations
        
        except Exception as e:
            logger.error(f"Failed to export Neo4j graph store: {str(e)}")
            raise