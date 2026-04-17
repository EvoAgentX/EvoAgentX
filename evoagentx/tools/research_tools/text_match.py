import re
import math
import spacy
import unicodedata
import pickle
from pathlib import Path
from typing import List, Tuple
from collections import Counter, defaultdict

# -----------------------------
# Normalization (accent + greek + basic cleanup + lemmatization)
# -----------------------------
try:
    spacy_disabled_functions = ["parser", "ner", "textcat"]
    nlp = spacy.load("en_core_web_sm", disable=spacy_disabled_functions) 
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=spacy_disabled_functions)

GREEK_MAP = {
    # basic letters (lower)
    "α":"alpha","β":"beta","γ":"gamma","δ":"delta","ε":"epsilon","ζ":"zeta","η":"eta","θ":"theta",
    "ι":"iota","κ":"kappa","λ":"lambda","μ":"mu","ν":"nu","ξ":"xi","ο":"omicron","π":"pi","ρ":"rho",
    "σ":"sigma","ς":"sigma","τ":"tau","υ":"upsilon","φ":"phi","χ":"chi","ψ":"psi","ω":"omega",

    # basic letters (upper)
    "Α":"alpha","Β":"beta","Γ":"gamma","Δ":"delta","Ε":"epsilon","Ζ":"zeta","Η":"eta","Θ":"theta",
    "Ι":"iota","Κ":"kappa","Λ":"lambda","Μ":"mu","Ν":"nu","Ξ":"xi","Ο":"omicron","Π":"pi","Ρ":"rho",
    "Σ":"sigma","Τ":"tau","Υ":"upsilon","Φ":"phi","Χ":"chi","Ψ":"psi","Ω":"omega",

    # common variant symbols
    "ϵ":"epsilon",      # lunate epsilon
    "϶":"epsilon",      # (rare) reversed lunate epsilon
    "ϑ":"theta",        # theta symbol
    "ϕ":"phi",          # phi symbol
    "ϖ":"pi",           # pi symbol
    "ϱ":"rho",          # rho symbol
    "ϰ":"kappa",        # kappa symbol
    "ϲ":"sigma",        # lunate sigma
    "ϒ":"upsilon",      # upsilon (symbol-ish)
    "ϴ":"theta",        # theta (symbol/capital)
}

LATEX_GREEK = {
    # lowercase
    "alpha":"alpha","beta":"beta","gamma":"gamma","delta":"delta","epsilon":"epsilon","varepsilon":"epsilon",
    "zeta":"zeta","eta":"eta","theta":"theta","vartheta":"theta","iota":"iota","kappa":"kappa","varkappa":"kappa",
    "lambda":"lambda","mu":"mu","nu":"nu","xi":"xi","omicron":"omicron","pi":"pi","varpi":"pi","rho":"rho","varrho":"rho",
    "sigma":"sigma","varsigma":"sigma","tau":"tau","upsilon":"upsilon","phi":"phi","varphi":"phi","chi":"chi","psi":"psi","omega":"omega",
    # uppercase
    "Gamma":"gamma","Delta":"delta","Theta":"theta","Lambda":"lambda","Xi":"xi","Pi":"pi","Sigma":"sigma","Upsilon":"upsilon","Phi":"phi","Psi":"psi","Omega":"omega",
}

_LATEX_RE = re.compile(r"\\([A-Za-z]+)\b")

def replace_latex_greek(s: str) -> str:
    def repl(m: re.Match) -> str:
        name = m.group(1)
        return " " + LATEX_GREEK.get(name, name) + " "
    return _LATEX_RE.sub(repl, s)

def replace_greek(s: str) -> str:
    s = replace_latex_greek(s)
    out = []
    for ch in s:
        if ch in GREEK_MAP:
            out.append(" " + GREEK_MAP[ch] + " ")
        else:
            out.append(ch)
    return "".join(out)

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s

def normalize_text(s: str) -> str:
    if not s: 
        return "" 
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = strip_accents(s) 
    s = replace_greek(s) 
    s = re.sub(r"[\u2010-\u2015\u2212\-]+", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    if not s:
        return [] 
    doc = nlp(s) 
    return [token.lemma_ for token in doc]

def normalize_text_lemmatized(s: str) -> str:
    tokens = tokenize(s)
    return " ".join(tokens)


# -----------------------------
# Recall & Maximum Segment Coverage
# -----------------------------

def token_recall(query_tokens: List[str], text_tokens: List[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    query_token_set = set(query_tokens)
    text_token_set = set(text_tokens)
    matched_tokens = query_token_set.intersection(text_token_set)
    recall = len(matched_tokens) / len(query_token_set)
    return recall

def token_precision(query_tokens: List[str], text_tokens: List[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    query_token_set = set(query_tokens)
    text_token_set = set(text_tokens)
    matched_tokens = query_token_set.intersection(text_token_set)
    precision = len(matched_tokens) / len(text_token_set)
    return precision

def build_text_ngram_sets(text_tokens: List[str], max_len: int) -> List[set]:
    # returns ngram_sets[L] = set of tuple ngrams of length L (1..max_len), index 0 is unused 
    n = len(text_tokens)
    ngram_sets = [set() for _ in range(max_len + 1)]
    for L in range(1, max_len + 1):
        if n < L:
            continue
        for i in range(0, n - L + 1):
            ngram_sets[L].add(tuple(text_tokens[i:i+L]))
    return ngram_sets

def max_segment_coverage(query: str, text: str, min_seg_len: int = 2, max_seg_len: int = 5, segment_penalty: float = 0.0) -> float: 
    # NOTE: text is expected not to be very long (e.g., title or short abstract)
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    num_query_tokens = len(query_tokens)

    if not query_tokens or not text_tokens:
        return 0.0 
    
    effective_max_len = min(max_seg_len, len(text_tokens), num_query_tokens) 
    text_ngram_sets = build_text_ngram_sets(text_tokens, effective_max_len)
    matches: List[List[int]] = [[] for _ in range(num_query_tokens)]
    for i in range(num_query_tokens):
        for L in range(min(effective_max_len, num_query_tokens-i), min_seg_len-1, -1):
            if tuple(query_tokens[i:i+L]) in text_ngram_sets[L]:
                matches[i].append(L)
    
    dp_score = [0.0] * (num_query_tokens + 1)
    dp_coverage = [0] * (num_query_tokens + 1) 
    dp_segment = [0] * (num_query_tokens + 1) 

    for i in range(num_query_tokens - 1, -1, -1):
        best_score = dp_score[i + 1] 
        best_coverage = dp_coverage[i + 1]
        best_segment = dp_segment[i + 1]
        for L in matches[i]:
            j = i + L 
            take_coverage = dp_coverage[j] + L 
            take_segment = dp_segment[j] + 1 
            take_score = take_coverage - segment_penalty * max(0, take_segment - 1)
            if take_score > best_score:
                best_score, best_coverage, best_segment = take_score, take_coverage, take_segment 
        dp_score[i], dp_coverage[i], dp_segment[i] = best_score, best_coverage, best_segment
    
    coverage_ratio = dp_coverage[0] / num_query_tokens
    return coverage_ratio

def text_match(
    query: str,
    text_candidates: List[str],
    recall_weight: float = 0.6,
    coverage_weight: float = 0.4,
    recall_threshold: float = 0.55,
    precision_threshold: float = 0.4, 
    coverage_threshold: float = 0.35,
    score_threshold: float = 0.55,
    min_seg_len: int = 2,
    max_seg_len: int = 5,
    segment_penalty: float = 0.15,
) -> dict:
    """
    Rank candidate texts against the query and decide whether the top candidate matches.

    Args:
        query: Reference text that we attempt to match.
        text_candidates: Iterable of candidate texts, each candidate text is expected to be short (e.g., title or short abstract). 
        recall_weight: Weight of recall in the final score.
        coverage_weight: Weight of max segment coverage in the final score.
        recall_threshold: Minimum recall required for a match decision.
        precision_threshold: Minimum precision required for a match decision.
        coverage_threshold: Minimum coverage required for a match decision.
        score_threshold: Minimum combined score required for a confident match.
        min_seg_len: Minimum segment length passed to max_segment_coverage.
        max_seg_len: Maximum segment length passed to max_segment_coverage.
        segment_penalty: Penalty factor passed to max_segment_coverage.

    Returns:
        A dictionary containing the match decision, the best candidate (if any),
        metrics of the best candidate, and the ranked list of candidates.
    """
    if not query or not text_candidates:
        return {"match": False, "best_candidate": None, "best_candidate_metrics": None, "ranked_candidates": []}

    query_tokens = tokenize(query)
    if not query_tokens:
        return {"match": False, "best_candidate": None, "best_candidate_metrics": None, "ranked_candidates": []}

    ranked = []
    for idx, candidate in enumerate(text_candidates):

        candidate_tokens = tokenize(candidate)
        if not candidate_tokens:
            # for empty candidate, assign zero scores 
            ranked.append({"index": idx, "text": candidate, "recall": 0.0, "segment_coverage": 0.0, "score": 0.0})
            continue 
        
        recall = token_recall(query_tokens, candidate_tokens)
        precision = token_precision(query_tokens, candidate_tokens)
        coverage = max_segment_coverage(
            query,
            candidate,
            min_seg_len=min_seg_len,
            max_seg_len=max_seg_len,
            segment_penalty=segment_penalty,
        )
        score = recall_weight * recall + coverage_weight * coverage
        ranked.append(
            {
                "index": idx,
                "text": candidate,
                "recall": recall,
                "precision": precision,
                "segment_coverage": coverage,
                "score": score,
            }
        )

    if not ranked:
        return {"match": False, "best_candidate": None, "best_candidate_metrics": None, "ranked_candidates": []}

    ranked.sort(key=lambda x: (x["score"], x["segment_coverage"], x["recall"]), reverse=True)
    best = ranked[0]

    match_decision = (
        best["score"] >= score_threshold and
        (best["recall"] >= recall_threshold or best["segment_coverage"] >= coverage_threshold) and 
        best["precision"] >= precision_threshold 
    )

    result = {
        "match": match_decision,
        "best_candidate": best["text"] if match_decision else None,
        "best_candidate_metrics": {
            "index": best["index"],
            "score": best["score"],
            "recall": best["recall"],
            "precision": best["precision"],
            "segment_coverage": best["segment_coverage"]
        },
        "ranked_candidates": ranked,
    }
    return result


def jaccard_similarity(query_tokens: List[str], text_tokens: List[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    query_token_set = set(query_tokens)
    text_token_set = set(text_tokens)
    intersection = query_token_set.intersection(text_token_set)
    union = query_token_set.union(text_token_set)
    return len(intersection) / len(union) 


# -----------------------------
# Simple BM25 Similarity 
# -----------------------------

class BM25: 

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):

        self.corpus = corpus 
        self.k1 = k1 
        self.b = b 

        self.docs = [tokenize(doc) for doc in corpus] 
        self.N = len(self.docs) 
        self.doc_lens = [len(doc) for doc in self.docs] 
        self.avg_doc_len = sum(self.doc_lens) / max(self.N, 1) 

        self.tf = [Counter(doc) for doc in self.docs] 
        self.df = defaultdict(int) 
        for doc_tf in self.tf:
            for term in doc_tf:
                self.df[term] += 1 
    
    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def _calculate_bm25_score(self, query_tokens: List[str], doc_tf: Counter, doc_len: int) -> float:
        """
        Calculate BM25 score given query tokens, document term frequencies and document length.

        Args:
            query_tokens: Tokenized query
            doc_tf: Term frequency counter for the document
            doc_len: Length of the document

        Returns:
            BM25 score
        """
        score = 0.0
        for term in query_tokens:
            if term not in doc_tf:
                continue
            tf = doc_tf[term]
            idf = self._idf(term)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def _score_with_ith_doc(self, idx: int, query_tokens: List[str]) -> float:
        """
        Calculate BM25 score between query and the i-th document in the corpus.

        Args:
            idx: Index of the document in the corpus
            query_tokens: Tokenized query

        Returns:
            BM25 score
        """
        if not query_tokens:
            return 0.0
        doc_tf = self.tf[idx]
        doc_len = self.doc_lens[idx]
        return self._calculate_bm25_score(query_tokens, doc_tf, doc_len)

    def score(self, query: str, text: str) -> float:
        query_tokens = tokenize(query)
        if not query_tokens:
            return 0.0
        try:
            idx = self.corpus.index(text)
            return self._score_with_ith_doc(idx, query_tokens)
        except ValueError:
            doc_tokens = tokenize(text)
            doc_tf = Counter(doc_tokens)
            doc_len = len(doc_tokens)
            return self._calculate_bm25_score(query_tokens, doc_tf, doc_len)

    def rank(self, query: str, topk: int = None) -> List[Tuple[int, float]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            raise ValueError("input `query` to BM25.rank is empty. Fail to rank documents.")
        results = [(i, self._score_with_ith_doc(i, query_tokens), self.corpus[i]) for i in range(self.N)]
        results.sort(key=lambda x: x[1], reverse=True)
        if topk:
            results = results[:topk]
        return results

    def save_index(self, filepath: str) -> None:
        """
        Save the BM25 index to a file for later loading.

        Args:
            filepath: Path where the index should be saved
        """
        index_data = {
            'corpus': self.corpus,
            'k1': self.k1,
            'b': self.b,
            'docs': self.docs,
            'N': self.N,
            'doc_lens': self.doc_lens,
            'avg_doc_len': self.avg_doc_len,
            'tf': self.tf,
            'df': dict(self.df),  # Convert defaultdict to regular dict for pickling
        }

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_index(cls, filepath: str) -> 'BM25':
        """
        Load a BM25 index from a saved file.

        Args:
            filepath: Path to the saved index file

        Returns:
            BM25 instance with loaded index
        """
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)

        # Create instance without going through __init__
        instance = cls.__new__(cls)

        # Restore all attributes
        instance.corpus = index_data['corpus']
        instance.k1 = index_data['k1']
        instance.b = index_data['b']
        instance.docs = index_data['docs']
        instance.N = index_data['N']
        instance.doc_lens = index_data['doc_lens']
        instance.avg_doc_len = index_data['avg_doc_len']
        instance.tf = index_data['tf']
        instance.df = defaultdict(int, index_data['df'])
        instance._idf_cache = {}

        return instance
