import copy
import regex
import string
import unicodedata
from typing import List
from collections import Counter
from ..core.logging import logger

#--------------------------
# QA Metrics
#--------------------------

# Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s : str) -> str:
    """Normalize an answer string for consistent evaluation.
    
    Applies a series of transformations to standardize answer strings:
    1. Removes articles ('a', 'an', 'the')
    2. Removes punctuation
    3. Fixes whitespace
    4. Converts to lowercase
    
    This normalization helps in comparing answers that are semantically 
    equivalent but have different surface forms.
    
    Args:
        s: The answer string to normalize
        
    Returns:
        The normalized answer string
    """
    def remove_articles(text: str) -> str:
        """Remove articles ('a', 'an', 'the') from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with articles removed
        """
        return regex.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text: str) -> str:
        """Standardize whitespace in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with standardized whitespace
        """
        return ' '.join(text.split())
    
    def remove_punc(text: str) -> str:
        """Remove punctuation from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with punctuation removed
        """
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def exact_match_score(prediction : str, ground_truth : str) -> float:
    """Calculate exact match score between prediction and ground truth.
    
    The exact match score is 1.0 if the normalized prediction exactly matches
    the normalized ground truth, and 0.0 otherwise.
    
    Args:
        prediction: The predicted answer
        ground_truth: The correct answer
        
    Returns:
        1.0 for exact match after normalization, 0.0 otherwise
        
    Raises:
        AssertionError: If ground_truth is not a string
    """
    assert isinstance(ground_truth, str), f"ground_truth must be a string, but got {type(ground_truth)}"
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def ems(prediction : str, ground_truths : List[str]) -> float:
    """Calculate exact match score against multiple ground truths.
    
    Returns the maximum exact match score against any of the ground truth answers.
    
    Args:
        prediction: The predicted answer
        ground_truths: List of acceptable ground truth answers
        
    Returns:
        The maximum exact match score (1.0 if the prediction matches any ground truth, 0.0 otherwise)
        
    Raises:
        AssertionError: If ground_truths is not a list
    """
    assert isinstance(ground_truths, list), f"ground_truths must be a list, but got {type(ground_truths)}"
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

# F1 Evaluation from HotPotQA evaluation script: https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py
def f1_score(prediction : str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth.
    
    F1 score is the harmonic mean of precision and recall, based on the overlap 
    of tokens between the prediction and ground truth answers.
    
    Special cases for yes/no/noanswer questions are handled separately.
    
    Args:
        prediction: The predicted answer
        ground_truth: The correct answer
        
    Returns:
        F1 score between 0.0 and 1.0
        
    Raises:
        AssertionError: If ground_truth is not a string
    """
    assert isinstance(ground_truth, str), f"ground_truth must be a string, but got {type(ground_truth)}"
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    ZERO_METRIC = (0, 0, 0)
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC[0]
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC[0]
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC[0]
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    # return f1, precision, recall
    return f1


def _normalize(text):
    """Apply Unicode normalization to text.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text with consistent Unicode representation
    """
    return unicodedata.normalize('NFD', text)

class Tokenizer(object):
    """Base tokenizer class.
    
    Abstract base class that defines the interface for tokenizers.
    Tokenizers implement tokenize, which should return a Tokens class.
    """
    def tokenize(self, text):
        """Tokenize text into a Tokens object.
        
        Args:
            text: Text to tokenize
            
        Returns:
            A Tokens object
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError

    def shutdown(self):
        """Clean up resources used by the tokenizer."""
        pass

    def __del__(self):
        """Destructor to ensure proper resource cleanup."""
        self.shutdown()


class Tokens(object):
    """A class to represent a list of tokenized text.
    
    This class provides various methods to access and manipulate
    tokens, including their text, offsets, parts of speech, etc.
    
    Attributes:
        TEXT: Index for token text
        TEXT_WS: Index for token text with whitespace
        SPAN: Index for token character span
        POS: Index for part-of-speech tag
        LEMMA: Index for lemmatized form
        NER: Index for named entity tag
    """

    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        """Initialize Tokens object.
        
        Args:
            data: List of token data tuples
            annotators: Set of annotation types available
            opts: Optional settings
        """
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens.
        
        Returns:
            Integer count of tokens
        """
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j).
        
        Args:
            i: Start index (inclusive)
            j: End index (exclusive)
            
        Returns:
            A new Tokens object with the sliced data
        """
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i:j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted).
        
        Returns:
            The reconstructed original text
        """
        return "".join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token.
        
        Args:
            uncased: Whether to lowercase the text
            
        Returns:
            List of token texts
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token.
        
        Returns:
            List of character span tuples
        """
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        
        Returns:
            List of POS tags, or None if this annotation was not included
        """
        if "pos" not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        
        Returns:
            List of lemmas, or None if this annotation was not included
        """
        if "lemma" not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        
        Returns:
            List of NER tags, or None if this annotation was not included
        """
        if "ner" not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.
        
        Args:
            n: Upper limit of ngram length
            uncased: Whether to lowercase text
            filter_fn: Function that takes an ngram list and returns
                a boolean indicating whether to keep it
            as_strings: Whether to return ngrams as strings instead of lists
                
        Returns:
            List of ngrams, either as strings or token span tuples
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [
            (s, e + 1)
            for s in range(len(words))
            for e in range(s, min(s + n, len(words)))
            if not _skip(words[s : e + 1])
        ]

        # Concatenate into strings
        if as_strings:
            ngrams = ["{}".format(" ".join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag.
        
        Returns:
            List of entity text and tag tuples, or None if NER not available
        """
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get("non_ent", "O")
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while idx < len(entities) and entities[idx] == ner_tag:
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class SimpleTokenizer(Tokenizer):
    """Simple regex-based tokenizer.
    
    This tokenizer splits text on regex boundaries, keeping alpha-numeric
    tokens together and separating other characters.
    
    Attributes:
        ALPHA_NUM: Regex pattern for alpha-numeric tokens
        NON_WS: Regex pattern for non-whitespace characters
    """
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self, **kwargs):
        """Initialize the SimpleTokenizer.
        
        Args:
            annotators: None or empty set (this tokenizer does not support annotations)
        """
        self._regexp = regex.compile(
            "(%s)|(%s)" % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE,
        )
        if len(kwargs.get("annotators", {})) > 0:
            logger.warning(
                "%s only tokenizes! Skipping annotators: %s" % (type(self).__name__, kwargs.get("annotators"))
            )
        self.annotators = set()

    def tokenize(self, text):
        """Tokenize text using regex patterns.
        
        Args:
            text: Text to tokenize
            
        Returns:
            A Tokens object containing the tokenized text
        """
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append(
                (
                    token,
                    text[start_ws:end_ws],
                    span,
                )
            )
        return Tokens(data, self.annotators)


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text.
    
    Args:
        text: The text to search
        pattern: The regex pattern to search for
        
    Returns:
        Boolean indicating whether the pattern was found
    """
    try:
        pattern = regex.compile(pattern, flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE)
    except BaseException:
        return False
    return pattern.search(text) is not None


# Acknowledgement: https://github.com/facebookresearch/DPR/blob/main/dpr/data/qa_validation.py#L175
def has_answer(answers, text, match_type="string") -> bool:
    """Check if the text contains any of the answer strings.
    
    This function supports two types of matching:
    1. String matching: Tokenizes both text and answers and checks for 
       token-level matches
    2. Regex matching: Treats answers as regex patterns to search in text
    
    Args:
        answers: List of possible answer strings
        text: Text to search for answers
        match_type: Type of matching to perform ("string" or "regex")
        
    Returns:
        Boolean indicating whether any answer was found in the text
    """
    text = _normalize(text)

    tokenizer = SimpleTokenizer()

    if match_type == "string":
        # Answer is a list of possible strings
        text = tokenizer.tokenize(text).words(uncased=True)

        for single_answer in answers:
            single_answer = _normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)

            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i : i + len(single_answer)]:
                    return True

    elif match_type == "regex":
        # Answer is a regex
        for single_answer in answers:
            single_answer = _normalize(single_answer)
            if regex_match(text, single_answer):
                return True
            
    return False

def acc_score(prediction : str, ground_truths : List[str]) -> float:
    """Calculate accuracy score based on string matching.
    
    Uses the has_answer function with string matching to determine
    if the prediction contains any of the ground truth answers.
    
    Args:
        prediction: The predicted answer
        ground_truths: List of acceptable ground truth answers
        
    Returns:
        1.0 if the prediction contains any ground truth, 0.0 otherwise
        
    Raises:
        AssertionError: If ground_truths is not a list
    """
    assert isinstance(ground_truths, list), f"ground_truths must be a list, but got {type(ground_truths)}"
    return float(has_answer(answers=ground_truths, text=prediction, match_type="string"))