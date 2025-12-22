# Borrow from llama_index
DEFAULT_SYNONYM_EXPAND_TEMPLATE = (
"Given some initial query, generate synonyms or related keywords up to {max_keywords} in total, "
"considering possible cases of capitalization, pluralization, common expressions, etc.\n"
"Provide all synonyms/keywords separated by '^' symbols: 'keyword1^keyword2^...'\n"
"Note, result should be in one-line, separated by '^' symbols."
"----\n"
"QUERY: {query_str}\n"
"----\n"
"KEYWORDS: "
)