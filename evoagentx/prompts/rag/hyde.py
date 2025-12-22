HYDE_SYSTEM_IMPLE_ = """You are a highly capable language model tasked with performing a Hypothetical Document Embedding (HyDE) transformation for a Retrieval-Augmented Generation (RAG) system. Your goal is to generate a concise hypothetical document that captures the semantic intent of the input query, which will be used to enhance retrieval."""
HYDE_IMPLE_ = """
**Instructions**:
1. The input query may be in any language. Analyze the language of the query and generate the hypothetical document in the *same language* as the query.
2. The hypothetical document should be a short, coherent passage (2-3 sentences) that answers or elaborates on the query as if it were a relevant document from a knowledge base.
3. Ensure the generated document is semantically rich, capturing the core intent and key details of the query.
4. Keep the tone neutral and informative, suitable for a retrieval context.

**Input Query**: {query}

**Output Format**:
[Your generated document in the query's language]
"""

DEFAULT_HYDE_PROMPT = HYDE_IMPLE_