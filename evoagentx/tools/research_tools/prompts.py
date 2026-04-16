BIB_NOT_FOUND_TEXT = "No BibTeX entry found for the given title or keyword."

DBLP_BIBTEXT_POLISHING_PROMPT = """
You are a BibTeX entry polishing expert. Your task is to polish a BibTeX entry to remove unnecessary fields and format the entry correctly. 

You should STRICTLY follow the following rules:
1. Rename the BibTeX entry tag to: the last name of the first author + the publication year + the first word of the paper title (if the first word contains special characters, like {{}}, :, - , etc, use the part before the special character). Note that the tag should be in lowercase.  
2. Normalize the author field by converting all author names into the "Lastname, Firstname" format, regardless of their original format. Ensure that each author follows this inverted order format and that authors are separated using 'and'. 
3. Remove the following fields: editor, publisher, url, doi, timestamp, biburl, bibsource. 
4. For conference or journal information fields (e.g., booktitle, journal), keep only the full official name of the conference or journal. 
    - If the field contains extra information such as dates, locations, or additional descriptors, remove everything except the complete conference/journal name. 
    - If the field contains only an abbreviation, convert the abbreviation to its full official name. 
    - For arxiv preprint papers, the journal field should be set to "arXiv preprint arXiv:XXX.XXX". 
5. Make the entire BibTeX entry more compact by removing unnecessary tabs, line breaks, and extra spaces, while keeping the formatting valid and readable. 
6. For the title field:
   - If there are bracket within the title, e.g., title={{REANO:} Constructing...}, you should remove the bracket and keep the content, i.e., "REANO: Constructing...".
   - Capitalization rules: 
      - In general, only capitalize the first letter of the title, and lowercase other words unless they fall under the exceptions below
      - If the title contains a colon, capitalize the fist word before AND after the colon, such as "REANO: Constructing..." -> "REANO: Constructing...". 
      - Preserve the original capitalization for: Acronyms, Proper nouns, Standardized model or dataset names 
      - Do NOT invent or guess which words should be capitalized beyond these rules. 
8. Only output the polished BibTeX entry, do not include any other text. 

IMPORTANT: 
- DO NOT invent any new fields or information. Only remove unnecessary fields and format the entry correctly. 
- If author name formatting needs to be modified, the transformation must strictly preserve both the original author order and the exact author names without introducing any changes to the spelling or sequence. 

### Examples 
The following are some examples of polished BibTeX entries: 

Example 1: 
@inproceedings{zhao2021multi,
  author={Zhao, Chen and Xiong, Chenyan and Boyd-Graber, Jordan and Daum{\'e} III, Hal},
  title={Multi-step reasoning over unstructured text with beam dense retrieval},
  booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={4635--4641},
  year={2021},
}

Example 2: 
@article{li2023pseudo,
  author={Li, Hang and Mourad, Ahmed and Zhuang, Shengyao and Koopman, Bevan and Zuccon, Guido},
  title={Pseudo relevance feedback with deep language models and dense retrievers: Successes and pitfalls},
  journal={ACM Transation on Information System},
  volume={41},
  number={3},
  pages={62:1--62:40},
  year={2023},
}

Example 3: 
@article{yue2025adenosine,
  title={Adenosine signalling drives antidepressant actions of ketamine and ECT},
  author={Yue, Chenyu and Wang, Na and Zhai, Haojiang and Yuan, Zhengwei and Cui, Yuting and Quan, Jing and Zhou, Yu and Fan, Xiaofeng and Wang, Hongshuang and Wu, Zhaofa and others},
  journal={Nature},
  pages={1--9},
  year={2025},
  publisher={Nature Publishing Group UK London}
}

Example 4: 
@article{chen2025learning,
  title={ReSearch: Learning to reason with search for LLMs via reinforcement learning},
  author={Chen, Mingyang and Sun, Linzhuang and Li, Tianpeng and Sun, Haoze and Zhou, Yijie and Zhu, Chenzheng and Wang, Haofen and Pan, Jeff Z and Zhang, Wen and Chen, Huajun and others},
  journal={arXiv preprint arXiv:2503.19470},
  year={2025}
}

### Inputs 
Polish the following BibTeX entry:
"""

AUTHOR_EXTRACTION_PROMPT = """
You are an expert publication metadata auditor. Extract the full list of author names for the referenced paper using ONLY the information provided below.

You MUST STRICTLY Output JSON with this exact schema:
{{"authors": ["Full Name 1", "Full Name 2"]}}

Hard rules:
1. Preserve the author order exactly as it appears on the web page.
2. Use the exact spelling shown on the page. Do not expand initials unless the page shows the full name.
3. Exclude affiliations, footnotes, and contributors who are not explicitly labeled as authors.
4. Try to find Normalize every author name into the "Lastname, Firstname" format (e.g., "Doe, John") while keeping the spelling identical to the source (only move the last name before the first name and insert a comma; never alter diacritics or introduce new text).
5. If you cannot confidently find all authors, return an empty list, i.e., {{"authors": []}}.

Paper title: "{title}"
Paper URL: {url}

Paper Page Content:
{context}
"""

PAPER_METADATA_EXTRACTION_PROMPT = """
You are an expert publication metadata extractor. Extract the paper title, full list of authors, abstract, PDF link, and venue information from the HTML page content provided below.

You MUST STRICTLY output JSON with this exact schema:
{{
    "title": "Paper Title", 
    "authors": ["Firstname Lastname", "Firstname Mid-name Lastname"],
    "abstract": "Abstract text or null",
    "pdf_link": "https://example.com/paper.pdf",
    "venue": "Conference/Journal Name or null",
    "is_published": true
}}

Hard rules:
1. Title: Extract the exact paper title as it appears on the page. Remove any extra formatting, line breaks, or special characters that are not part of the actual title.
2. Authors: 
   - Preserve the author order exactly as it appears on the web page.
   - Normalize every author name into the "Firstname Mid-name Lastname" format, where the middle name is optional (e.g., "John Doe", "Jane Smith", "John Michael Doe", "Mary Ann Johnson").
   - Use the exact spelling shown on the page. Do not expand initials unless the page shows the full name.
   - Exclude affiliations, footnotes, and contributors who are not explicitly labeled as authors.
   - If you cannot confidently find all authors, return an empty list, i.e., {{"title": "...", "authors": [], "abstract": "..."}}.
3. Abstract:
   - Extract the complete abstract content if available on the page.
   - If no abstract is found, set the value to null.
   - Format the abstract as a single paragraph, removing unnecessary line breaks that may not be intended (some web pages may have line breaks that are not desired).
   - Remove any HTML tags or extra formatting that may interfere with readability.
4. PDF Link:
   - Extract the BEST PDF download link from the page (e.g., links containing ".pdf", "download", "full text").
   - Return only ONE URL string. If no PDF link is found, return null.
   - Prioritize: direct PDF links > arXiv PDF links > publisher PDF links.
   - Look for buttons/links labeled "PDF", "Download PDF", "View PDF", etc.
5. Venue:
   - Extract the publication venue (conference or journal name) ONLY if the paper has been published.
   - Check for keywords like "Published in", "Proceedings of", "Journal:", "Conference:", etc.
   - If the paper is a preprint (arXiv, bioRxiv, etc.) or not yet published, set venue to null.
   - Set is_published to true if the paper has been formally published, false otherwise.
   - Common indicators of published papers: conference proceedings, journal names, DOI, publication dates.
   - Common indicators of unpublished papers: "preprint", "under review", "submitted", arXiv-only.
6. If the page content is not provided or does not contain any related paper information, you should output:
{{
    "title": null,
    "authors": [],
    "abstract": null,
    "pdf_link": null,
    "venue": null,
    "is_published": false
}}

Paper Page Content:
{context}
"""

PAPER_METADATA_EXTRACTION_PROMPT_NO_ABSTRACT = """
You are an expert publication metadata extractor. Extract ONLY the paper title and full list of authors from the HTML page content provided below. Do NOT extract the abstract.

You MUST STRICTLY output JSON with this exact schema:
{{
    "title": "Paper Title", 
    "authors": ["Firstname Lastname", "Firstname Mid-name Lastname"]
}}

Hard rules:
1. Title: Extract the exact paper title as it appears on the page. Remove any extra formatting, line breaks, or special characters that are not part of the actual title.
2. Authors: 
   - Preserve the author order exactly as it appears on the web page.
   - Normalize every author name into the "Firstname Mid-name Lastname" format, where the middle name is optional (e.g., "John Doe", "Jane Smith", "John Michael Doe", "Mary Ann Johnson").
   - Use the exact spelling shown on the page. Do not expand initials unless the page shows the full name.
   - Exclude affiliations, footnotes, and contributors who are not explicitly labeled as authors.
   - If you cannot confidently find all authors, return an empty list, i.e., {{"title": "...", "authors": []}}.
3. Abstract: Do NOT extract the abstract. Only focus on extracting the title and authors.
4. If the page content is not provided or does not contain any related paper information, you should output:
{{
    "title": null,
    "authors": []
}}

Paper Page Content:
{context}
"""

PAPER_BIBTEXT_FORMULATION_PROMPT = """
You are a BibTeX entry formulation expert. Your task is to formulate a proper BibTeX entry for LaTeX based on a Google Scholar Chicago format citation and an optional author list.

You should STRICTLY follow the following rules:

1. **Entry Type Selection**: Determine the correct BibTeX entry type (e.g., @article, @inproceedings, @book, @phdthesis, @mastersthesis, etc.) based on the Chicago citation format. Common types:
   - @article: for journal articles or arXiv preprint papers 
   - @inproceedings: for conference papers
   - @book: for books
   - @phdthesis: for PhD dissertations
   - @mastersthesis: for master's theses
   - @techreport: for technical reports
Do NOT guess beyond what is explicitly shown in the Chicago citation.

2. **Entry Tag**: Generate the BibTeX entry tag as: the last name of the first author (lowercase) + the publication year + the first word of the paper title (lowercase, if the first word contains special characters, use the part before the special character).
   - e.g., `zhao2021multi`

3. **Author Field Formatting**:
   - **Rule 3.1: If the authors list is NOT provided**:
     - Extract author information from the Chicago reference 
     - Format each author as "Lastname, Firstname" and separate authors using 'and' 
     - Example: `author={{Doe, John and Smith, Jane and Johnson, Mary Ann}}` 

   - **Rule 3.2: If the authors list is provided**:
     - If the authors list contains full names, e.g., ["John Doe", "Jane Smith", "Mary Ann Johnson"]: 
       - Use the provided author list directly 
       - Convert all author names into the "Lastname, Firstname" format and separate authors using 'and' 

     - If the authors list contains abbreviated names, e.g., [P Lewis, E Perez, A Piktus, F Petroni]:
       - Do NOT use the provided authors list 
       - Instead, extract author information from the Chicago reference
       - Format author names as "Lastname, Firstname" or "Lastname, Initial", depending on what the Chicago reference provides 
       - Append the following note at the end of the author field: "(Note that the author information might be incomplete)" 
       - Example: `author={{Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio (Note that the author information might be incomplete)}}`

4. **Field Extraction (STRICTLY FOLLOW THE CHICAGO CITATION)**: 
Extract ONLY what is explicitly present in the Chicago citation, including: 
   - title: The paper title (must include)
   - author: As specified in rule 3 (must include)
   - year: Publication year (must include)
   - journal: For journal articles or arXiv preprint papers (full official name)
   - booktitle: For conference papers (full official name)
   - pages: Page numbers if available
   - volume: Volume number if available
   - number: Issue number if available
   - publisher: Publisher name if available
   - address: Location if available
   - school: For theses
   - institution: For technical reports
If a field does not clearly appear in the Chicago citation, DO NOT include it.

5. Capitalization rules for the title field:
   - In general, only capitalize the first letter of the title, and lowercase other words unless they fall under the exceptions below
   - If the title contains a colon, capitalize the fist word before AND after the colon, such as "REANO: Constructing..." -> "REANO: Constructing...". 
   - Preserve the original capitalization for: Acronyms, Proper nouns, Standardized model or dataset names 
   - Do NOT invent or guess which words should be capitalized beyond these rules. 

6. **Field Formatting**:
   - Keep journal and booktitle fields as full official names (not abbreviations). If only the abbreviation is provided, convert the abbreviation to its full official name. 
   - Format page numbers using double hyphens (e.g., "4635--4641")
   - Remove unnecessary information from fields (e.g., dates, locations from booktitle/journal)

7. **CRITICAL CONSTRAINTS**:
   - DO NOT invent any new fields or information that is not present in the Chicago citation
   - DO NOT add fields like url, doi, timestamp, biburl, bibsource unless explicitly present in the Chicago citation
   - Only extract and format information that can be clearly identified from the provided Chicago citation
   - Preserve the exact spelling and content from the Chicago citation

8. **Output Format**: Output only the BibTeX entry, do not include any other text or explanations.
   - Do NOT add ```bibtext ...``` at your output. 
   - If you cannot formulate the BibTeX entry based on the Chicago citation, return the following error message: "Error: Failed to formulate the BibTeX entry based on the Chicago citation."

### Examples

Example 1 (with empty author list):
Chicago Citation: Zhao, Chen, Chenyan Xiong, Jordan Boyd-Graber, and Hal Daumé III. "Multi-step reasoning over unstructured text with beam dense retrieval." Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 4635-4641 (2021).
Authors: []

Output:
@inproceedings{{zhao2021multi,
  title={{Multi-step reasoning over unstructured text with beam dense retrieval}},
  author={{Zhao, Chen and Xiong, Chenyan and Boyd-Graber, Jordan and Daum{{\'e}} III, Hal}},
  booktitle={{Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies}},
  pages={{4635--4641}},
  year={{2021}},
}}

Example 2 (with author list from arXiv preprint paper):
Chicago Citation: Chen, Mingyang, Linzhuang Sun, Tianpeng Li et al. "Learning to reason with search for llms via reinforcement learning." arXiv preprint arXiv:2503.19470 (2025).
Authors: [Mingyang Chen, Linzhuang Sun, Tianpeng Li, Haoze Sun, Yijie Zhou] 

Output:
@article{{chen2025learning,
  title={{Learning to reason with search for llms via reinforcement learning}},
  author={{Chen, Mingyang and Sun, Linzhuang and Li, Tianpeng and Sun, Haoze and Zhou, Yijie}},
  journal={{arXiv preprint arXiv:2503.19470}},
  year={{2025}}
}}

Example 3 (with abbreviated authors list):
Chicago Citation: Lewis, Patrick, Ethan Perez, Aleksandra Piktus, Fabio Petroni et al. "Retrieval-augmented generation for knowledge-intensive nlp tasks." Advances in neural information processing systems 33 (2020): 9459-9474.
Authors: [P Lewis, E Perez, A Piktus, F Petroni]

Output:
@article{{lewis2020retrieval,
  title={{Retrieval-augmented generation for knowledge-intensive nlp tasks}},
  author={{Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio (Note that the author information might be incomplete)}},
  journal={{Advances in Neural Information Processing Systems}},
  volume={{33}},
  pages={{9459--9474}},
  year={{2020}},
}}

### Input

Chicago Citation:
{context}

Author List:
{authors}

Now formulate the BibTeX entry:
"""

ARXIV_PAPER_BIBTEXT_FORMULATION_PROMPT = """
You will be given the metadata of a single arXiv paper as a JSON object (Python dict-like), for example:

{
  "title": "title of the paper",
  "arxiv_id": "2504.01346v4",
  "authors": ["author1", "author2", "author3", ...],
  "links": {
    "html": "https://arxiv.org/abs/2504.01346v4",
    "pdf": "https://arxiv.org/pdf/2504.01346v4"
  },
  "url": "https://arxiv.org/abs/2504.01346v4",
  "published_date": "2025-04-02T04:24:41Z",
  "updated_date": "2025-10-05T07:24:41Z"
}

Your task is to convert this metadata into ONE valid BibTeX entry of type @article, similar to:

@article{chen2025learning,
  title={ReSearch: Learning to reason with search for LLMs via reinforcement learning},
  author={Chen, Mingyang and Sun, Linzhuang and Li, Tianpeng and Sun, Haoze and Zhou, Yijie and Zhu, Chenzheng and Wang, Haofen and Pan, Jeff Z and Zhang, Wen and Chen, Huajun and others},
  journal={arXiv preprint arXiv:2503.19470},
  year={2025}
}

Follow these STRICT rules:

1. General output requirements
   - Output ONLY the BibTeX entry.
   - Do NOT add any explanations, comments, natural language, markdown, or code fences.
   - The entry MUST start with "@article{" and end with the closing "}" on its own line.

2. Citation key (the part after "@article{")
   - Form: <first_author_lastname><year><first_word_of_title>, all lowercase, no spaces.
   - <first_author_lastname>:
       - Take the first author's family name (last token of the name string when split by space).
       - Example: "Mingyang Chen" -> "chen"; "Jiaru Zou" -> "zou".
   - <year>:
       - Use the year extracted from "published_date" (preferred) or, if missing, from "updated_date".
       - Dates are ISO strings like "2025-04-02T04:24:41Z"; extract the first 4 digits.
   - <first_word_of_title>:
       - Take the first word of the title, (if the first word contains special characters, use the part before the special character). Note that the tag should be in lowercase. 
   - Concatenate all three parts directly without separators, e.g. "chen2025learning".

3. title field
   - Capitalization rules:
      - In general, only capitalize the first letter of the title, and lowercase other words unless they fall under the exceptions below
      - If the title contains a colon, capitalize the fist word before AND after the colon, such as "REANO: Constructing..." -> "REANO: Constructing...". 
      - Preserve the original capitalization for: Acronyms, Proper nouns, Standardized model or dataset names 
      - Do NOT invent or guess which words should be capitalized beyond these rules. 
   - Format: title={...}

4. author field
   - Use the "authors" array.
   - Each element is "First Last" or "First Middle Last" etc.
   - Convert each author into "Last, First Middle" form:
       - Split by spaces; the last token is the family name; all preceding tokens together form the given name part.
       - Example: "Mingyang Chen" -> "Chen, Mingyang"; "Jeff Z Pan" -> "Pan, Jeff Z".
   - Join multiple authors with " and " (BibTeX standard).
   - If the list is very long, you MAY append " and others" at the end (optional, but allowed).

5. journal field
   - This MUST follow the pattern: journal={arXiv preprint arXiv:<ID>},
   - <ID> is the arXiv identifier WITHOUT any version suffix.
   - Determine the raw arXiv ID as follows (first non-empty source wins):
       1) Use the "arxiv_id" field if present.
       2) Otherwise, parse from "url" or "links.html" by taking the last path segment after "/abs/".
          Example: "https://arxiv.org/abs/2504.01346v4" -> "2504.01346v4".
   - Strip the version suffix: if the ID ends with "v" followed by digits (e.g., "v4", "v12"), remove that suffix.
       - Example: "2504.01346v4" -> "2504.01346".
   - Use the cleaned ID in the journal:
       - Example: journal={arXiv preprint arXiv:2504.01346},
   - NEVER keep the version string (e.g. "v4") in the journal field.

6. year field
   - As in the key: use the year from "published_date"; if missing, fall back to "updated_date".
   - Extract the first 4-digit year from the ISO timestamp.
   - Example: "2025-04-02T04:24:41Z" -> year={2025},

7. Do not include other fields in the BibTeX entry except the ones mentioned above. 

IMPORTANT: If you cannot formulate the BibTeX entry based on the metadata, return the following error message: "Error: Failed to formulate the BibTeX entry based on the metadata." and nothing else.

Now, given the input JSON, output ONLY the corresponding BibTeX @article entry following all rules above.
"""

EXTRACT_PAPER_METADATA_FROM_GOOGLE_SEARCH_RESULTS_PROMPT = """
You are an expert at extracting paper metadata from Google Search results.

**Task**: Given a user query (which may contain a paper title, partial title, or related keywords) and Google Search results, extract the most relevant paper's metadata.

**Selection Strategy**:
1. Find the item that best matches the user's query - the query may be:
   - A complete paper title
   - A partial paper title or main keywords from the title
   - Title with additional context (authors, year, venue, etc.)
   - Use semantic matching, not just exact string matching
2. **MANDATORY**: If ANY result contains an arXiv link (arxiv.org) or PubMed link (pubmed.ncbi.nlm.nih.gov or ncbi.nlm.nih.gov/pubmed), you MUST select that result, even if other results seem more relevant. This is a hard requirement.
3. If no arXiv/PubMed links exist, prioritize other openly accessible sources in this order:
   - Conference/journal open access: ACL Anthology, OpenReview, EMNLP, NeurIPS, ICLR, etc.
   - Semantic Scholar
   - University/institutional repositories (edu domains, institutional archives)
   - Other academic websites
4. Avoid paywalled sites when possible (e.g., IEEE Xplore, ACM Digital Library, Springer, Elsevier, Nature, Science, ScienceDirect - unless they explicitly provide open access).

**Input Format**:
- title: The user's input query (may contain title, partial title, or keywords)
- papers_info: A list of search results from Google, each containing fields like position, title, link, snippet, source, author, date, etc.

**Required Output Fields**:
1. **paper_title**: Infer the MOST COMPLETE and ACCURATE title by combining information from:
   - The user's query (may contain the full or partial title)
   - All search result titles (often truncated with "...")
   - Reconstruct the most reasonable complete title that matches the paper
   - If user query appears to be a complete title and matches well, prefer it
   - If search results provide more complete information, use that
   - Remove any truncation markers like "..." from the final title
2. **paper_link**: The 'link' field from the best matching item
3. **snippet**: The 'snippet' field from the best matching item
4. **year**: Extract publication year from ANY item in the results. Look in 'date' fields, snippets, or links. If not found, set to null.
5. **venue_info**: Extract venue information with the following priority:
   - If venue information is found in ANY result (conference/journal name), use it
   - Otherwise, use the 'source' field from the best matching item
   - If 'author' field exists in any result, format as: "author, venue"
   - If venue does not exist, set to null. 

**Output Format**:
Return a JSON object with these exact keys:
{{
    "paper_title": "Most complete inferred title (combining user query + search results)",
    "paper_link": "URL from best match",
    "snippet": "snippet from best match",
    "year": "extracted year or null",
    "venue_info": "extracted venue information or null"
}}

**Important Notes**:
- **CRITICAL**: If ANY result contains an arXiv or PubMed link, you MUST select it regardless of other factors. Check all results for "arxiv.org" or "pubmed.ncbi.nlm.nih.gov" or "ncbi.nlm.nih.gov/pubmed" in the link field.
- Search result titles are often truncated with "..." - use ALL available information to reconstruct the complete title
- Combine evidence from: user query, all search result titles, snippets, and metadata
- The user query may contain the full title, partial title, or additional context (authors, year, etc.)
- For year and venue_info, scan ALL items in papers_info, not just the best match
- Prefer academic sources (arXiv, PubMed, university sites, ACL, IEEE, ACM, etc.) over general websites
- Use semantic similarity for matching - key concepts matter more than exact wording
- If multiple results seem relevant, prefer the one with more complete metadata
- Your goal is to return the most accurate and complete title possible

**Output Format**:
- You MUST output ONLY the JSON object with the exact keys and values as specified above. DO NOT include any other text or explanations. 
- You MUST output all the required fields, even if they are null. Do NOT omit any fields. 
- If you cannot extract the paper metadata from the Google Search results, return the following error message: "Error: Failed to extract the paper metadata from the Google Search results." and nothing else. 

**Inputs**:
User Query: {title}

Google Search Results:
{papers_info}

Now, extract the paper metadata from the Google Search results and output the JSON object:
"""

QUERY_EXPANSION_FOR_RECALL_PROMPT = """
You are an expert in academic information retrieval and query expansion for HIGH RECALL while PRESERVING the core concept. 

**Task**: 
Given a user query, generate up to {max_queries} alternative academic search queries that help retrieve more relevant or seminal/foundational papers. 

**Core Preservation Rule (mandatory)**:
- Each output query MUST explicitly contain the original core term(s) from the user query OR a widely-used near-equivalent head term for the SAME concept.

**Important Constraints**:
- Do NOT generate trivial variants such as: hyphenation changes, capitalization changes, plural/singular changes
- Do NOT output the original query verbatim.
- Acronyms alone (e.g., "RAG") are NOT sufficient unless paired with a descriptive phrase.
- Every query must add REAL retrieval value beyond the original wording.

**What to Generate**:
- Queries that are likely to retrieve **seminal / foundational / canonical papers** on the topic are preferred if possible.
- Add a commonly-used modifier to the core head term (e.g., self-*, evolving, adaptive, continual).
- Add a standard nearby phrase that co-occurs in titles/abstracts with the core head term.
- Add a mechanism-level descriptor tightly tied to the core concept (e.g., online learning, continual adaptation), BUT keep the core head term present.

**Rules**:
- Each query must be a natural academic search phrase
- Each query should be no longer than 8 words
- Use only well-established terminology (no invented phrases)
- Do NOT add years, authors, datasets, or venues
- Do NOT include explanations or formatting

**Output Format**: 
- Output ONLY the queries, one per line. No numbering, no quotes, no explanations. 
- If you cannot generate any alternative queries, output 'Error: Failed to generate alternative queries.' and nothing else. 

**Input**:
User Query: {query}

Now, generate up to {max_queries} alternative queries for the user query:
"""


SEMINAL_PAPERS_FOR_TOPIC_PROMPT = """
You are an expert research assistant and science historian.

**Task**: Given an academic topic, identify up to {max_papers} seminal / foundational / canonical papers that are central to this topic. 

**How to Think (internal reasoning)**: 
- Briefly decompose the topic into key sub-questions or sub-areas. 
- Recall historically important milestones, paradigm-shifting ideas, and standard reference works (including classic surveys or monographs if relevant). 
- Prefer works that are: 
   - Widely cited OR widely taught as "the" reference, 
   - Introduced a core concept, model, or framework, 
   - Or consolidated and shaped the field (e.g., influential surveys or benchmark papers). 
   
**Selection Rules**: 
- Stay strictly within the scope of the given topic; avoid generic ML/NLP/CS classics unless they are central to this topic. 
- Prefer diversity across time: 
   - Early foundational papers that introduced the idea. 
   - Follow-up or consolidation works that defined the modern formulation.

**Output Format**:
- You MUST output ONLY a JSON array with the following schema:
```json
[
  {{
    "title": "Exact or best-guess paper title",
    "year": 2020,                // integer year or null if unknown
    "role": "foundational | survey | benchmark | breakthrough | other", 
  }},
  ...
]
```
- Do NOT include any other text, thoughts, explanations or comments.
- If you cannot identify any reasonable seminal papers, return an empty JSON array []. 

**Input**:
Topic: {topic}

Now generate up to {max_papers} seminal papers for the topic and output the JSON array:
"""



MATCH_CITATIONS_PROMPT = """
You are an expert in matching in-text citations to references in a paper.

**Task**:
You will be given: 1) in-text citations from a paper; 2) the paper's References section text.
Your task is to identify which reference entries the citations refer to. Return the matched paper titles and years.

Important requirements:
- You should try to match each citation to the corresponding reference entry. Do not skip any citation.
- If the citation is numeric like "[1]" or "[12]", match it to the corresponding numbered reference if possible.
- If the citation is author-year like "Author et al. (2023)", match it to the corresponding reference if possible.
- The "title" MUST be copied as closely as possible from the References entry (same wording/capitalization if present).
   - You should use ONLY the provided References text. DO NOT invent papers.
- The "year" MUST be ONLY the 4-digit year, i.e., YYYY. If the citation/ref shows "2023a" or "2023b", output "2023".
- It is NORMAL that some citations cannot be matched to any reference entry. In this case, you should ignore these citations and focus on the ones that can be matched.

**Output format**:
Return ONLY a valid JSON array like:
```json
[
   {{
      "title":"...",
      "year":"YYYY"
   }},
   ...
]
```
- Only output the JSON array, DO NOT include any other text or explanations.
- If none of the citations can be matched to any reference entry, return an empty JSON array [].
- For other cases where you cannot finish the task, still return an empty JSON array [].

In-text citation:
{citations}

References text:
{references}
"""

DBLP_VENUE_SELECTION_PROMPT = """Given the user's venue query: "{venue}"

Here are the matching venues found:
{venue_list}

Please identify which venue best matches the user's query. Return your answer in JSON format:
{{
    "selected_index": <index number starting from 1>,
    "venue_name": "<full venue name>",
    "url": "<venue URL>"
}}

Only return the JSON, nothing else.

If no venue matches the user's query or you cannot determine the best match, return an empty JSON object: {{}}
"""

DBLP_YEAR_FILTER_RANGE_PROMPT = """Given the following list of conferences/journals:

{conference_list}

The user wants conferences/journals {year_constraint}. Please identify which conferences/journals match this year range based on their titles.

Return your answer in JSON format:
{{
    "selected_indices": [<list of indices, 1-based>]
}}

You should rank the selected conferences/journals in descending order by year (latest first).

Only return the JSON, nothing else.

If no conferences/journals match the year constraint or you cannot determine any matches, return an empty JSON object: {{}}
"""


EXTRACT_AUTHORS_FROM_CHICAGO_CITATION_PROMPT = """
You are an expert at extracting author names from Chicago-style citations.

**Task**: Given a Chicago-style citation, extract all author names and return them as a list.

**Input Format**:
- chicago_citation: A citation string in Chicago format, e.g.:
  "Smith, John A., Mary B. Johnson, and Robert C. Williams. "Paper Title." Journal Name 42, no. 3 (2020): 123-145."

**Rules**:
1. Extract all author names from the citation.
2. Convert each author name to "Firstname Lastname" format (not "Lastname, Firstname").
3. Handle "et al." by returning the extracted authors with a flag indicating the list is incomplete.
4. If only one author is present, still return a list with one element.
5. Handle various author formats:
   - "Smith, John" -> "John Smith"
   - "John Smith" -> "John Smith"
   - "Smith, John A." -> "John A. Smith"
   - Multiple authors separated by ", and " or " and "

**Output Format**:
Return a JSON object with the following schema:
{{
    "authors": ["Firstname Lastname", "Firstname Lastname", ...],
    "is_complete": true/false
}}

- "is_complete" should be false if the citation contains "et al." or similar indicators that not all authors are listed.
- If you cannot extract any authors, return: {{"authors": [], "is_complete": false}}

**Input**:
Chicago Citation: {chicago_citation}

Now extract the authors and output the JSON:
"""


INFER_PAPER_VENUE_PROMPT = """
You are an expert at identifying academic publication venues.

**Task**: Given paper metadata with potentially noisy or incomplete venue information, infer the most appropriate venue identifier based ONLY on the provided data.

**Input**:
- Paper Title: {paper_title}
- Year: {year}
- Noisy Venue Info: {venue_info}
- Paper Link: {paper_link}

**Rules**:
1. Analyze ONLY the provided information - do not make assumptions beyond what's given.
2. Priority order for venue identification:
   a) If a specific conference/workshop acronym is identifiable → return it (e.g., "NeurIPS", "CVPR", "ACL")
   b) If a journal name is identifiable → return it (e.g., "Nature", "JMLR", "TACL")
   c) If only a publishing platform is identifiable → return the platform (e.g., "IEEE", "ACM", "Springer", "arXiv")
   d) If the paper link is from arXiv → return "arXiv"
3. Examples:
   - "Y Dong, S Wang - 2024 5th ... - ieeexplore.ieee.org" → venue: "IEEE"
   - "J Chen - NeurIPS 2024 - proceedings.neurips.cc" → venue: "NeurIPS"
   - "A Smith - Nature Machine Intelligence, 2024" → venue: "Nature Machine Intelligence"
4. If no venue information can be extracted, set venue to null.

**Output Format**:
Return a JSON object with the following schema:
{{
    "venue": "venue name/platform or null",
    "confidence": "high/medium/low"
}}

Now infer the venue and output the JSON:
"""


EXTRACT_PAPER_METADATA_FROM_PDF_FIRST_PAGE_PROMPT = """
You are an expert at extracting paper metadata from the first page of academic papers.

**Task**: Given the text content from the first page of a PDF, extract the paper title, authors, and abstract.

**Input**:
{pdf_content}

**Rules**:
1. Title: Extract the main paper title. It's usually the largest/most prominent text near the top.
2. Authors:
   - Extract all author names in order.
   - Normalize to "Firstname Lastname" format.
   - Exclude affiliations, emails, and footnote markers.
   - If you cannot confidently extract all authors, return an empty list.
3. Abstract:
   - Extract the abstract content if present.
   - The abstract is usually labeled with "Abstract" and appears after the authors.
   - Remove the "Abstract" label itself.
   - If no abstract is found, return null.

**Output Format**:
Return a JSON object with the following schema:
{{
    "title": "Paper Title",
    "authors": ["Firstname Lastname", ...],
    "abstract": "Abstract text or null"
}}

If you cannot extract any information, return:
{{
    "title": null,
    "authors": [],
    "abstract": null
}}

Now extract the metadata and output the JSON:
"""