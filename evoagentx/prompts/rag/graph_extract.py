ENTITY_EXTRACT_PROMPT = """
Extract entities and their types from the following text. An entity is a specific, named object, such as a person, organization, location, or concept. The type should be a concise category (e.g., Person, Organization, Location, Concept). 
Return the results in JSON format as a list of objects, each with 'name' and 'type' fields.
Ensure the entities are distinct and relevant to the text's main topics.

***DO NOT*** answer the question itself if the given text is a question.

Text:
{text}

----------------
EXAMPLE:
Text: Tim Cook, CEO of Apple Inc., announced the new Apple Watch that monitors heart health. 
"UC Berkeley researchers studied the benefits of apples.\n"

Output:
```json
{"entities":[
    {'name': 'Tim Cook', 'type': 'PERSON',},
    {'name': 'Apple Inc.', 'type': 'COMPANY'},
    {'name': 'Apple Watch', 'type': 'PRODUCT'},
    {'name': 'heart health', 'type': 'HEALTH_METRIC'},
    {'name': 'UC Berkeley', 'type': 'UNIVERSITY'},
    {'name': 'benefits of apples', 'type': 'RESEARCH_TOPIC'}]
}
```
----------------

Output JSON:
```json
{"entities": [{"name": "entity_name", "type": "entity_type"}, ...]}
```
"""

RELATION_EXTRACT_PROMPT = """
Given the following text and a list of extracted entities, identify explicit, directed relationships between pairs of entities. A relationship should describe a clear, meaningful connection (e.g., "works_for", "located_in", "founded"). Return the results in JSON format as a list of triplets, each with 'source', 'target', and 'relation' fields. The 'source' and 'target' must match entity names from the provided list. 
Ensure relationships are specific and grounded in the text.

***DO NOT*** answer the question itself if the given text is a question.

Text:
{text}

Entities:
{entities_json}

Output JSON:
```json
{"graph": [{"source": "entity1", "target": "entity2", "relation": "relationship_label"}, ...]}
```
"""