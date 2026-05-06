import inspect
import re
from typing import Any, List, Optional, get_origin

ALLOWED_TYPES = ["string", "number", "integer", "boolean", "object", "array"]
URL_CHAR = r'[^\s,;"\'`<>\\\[\]\(\)\{\}。，；：！？""'']'


def _split_words(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    parts: List[str] = []
    for token in s.split():
        pieces = re.findall(r"[A-Z]+(?=[A-Z][a-z]|[0-9])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", token)
        parts.extend(pieces)
    return [p for p in parts if p]


def _strip_suffix(s: str, suffixes: List[str] = ["tool", "toolkit"]) -> str:
    s_clean = (s or "").strip()
    for suf in suffixes:
        if s_clean.lower().endswith(suf):
            return s_clean[: -len(suf)]
    return s_clean


def normalize_component_name(name: str, suffix: str) -> str:
    base = _strip_suffix(name)
    words = _split_words(base)
    normalized = "".join(w.capitalize() for w in words)
    if suffix not in ("Tool", "Toolkit"):
        suffix = "Tool"
    if not normalized:
        return suffix
    return f"{normalized}{suffix}"


def is_valid_description(desc: Optional[str], min_len: int = 10) -> bool:
    if desc is None:
        return False
    text = desc.strip()
    return len(text) >= min_len


TYPE_MAP = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
}
ORIGIN_MAP = {
    list: "array",
    tuple: "array",
    set: "array",
    dict: "object",
}
DOC_TYPE_MAP = {
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "double": "number",
    "number": "number",
    "str": "string",
    "string": "string",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "tuple": "array",
    "set": "array",
    "array": "array",
    "dict": "object",
    "mapping": "object",
    "object": "object",
}


def map_doc_type(label: str) -> str:
    base_match = re.match(r"([A-Za-z_]+)", label.strip())
    base = base_match.group(1).lower() if base_match else label.strip().lower()
    return DOC_TYPE_MAP.get(base, "string")


def _map_type_to_schema(py_type: Any) -> str:
    if py_type is None or py_type is inspect._empty:
        return "string"
    origin = get_origin(py_type)
    if origin in ORIGIN_MAP:
        return ORIGIN_MAP[origin]
    if isinstance(py_type, type) and py_type in TYPE_MAP:
        return TYPE_MAP[py_type]
    return "string"


def extract_descriptions(func):
    doc = inspect.getdoc(func) or ""
    if not doc.strip():
        return "", {}, [], []

    def collapse(parts_or_text):
        if isinstance(parts_or_text, str):
            parts = parts_or_text.splitlines()
        else:
            parts = parts_or_text
        return re.sub(r"\s+", " ", " ".join(p.strip() for p in parts if p is not None and p.strip())).strip()

    tokens = re.split(r"^\s*(Args|Arguments|Parameters|Returns|Required|Raises|Examples|Yields):\s*$", doc, flags=re.MULTILINE)
    preface = tokens[0] if tokens else doc
    func_desc = collapse(preface)

    sections = {}
    for i in range(1, len(tokens), 2):
        label = tokens[i].strip().lower()
        content = tokens[i + 1]
        sections[label] = content

    param_content = None
    for key in ("args", "arguments", "parameters"):
        if key in sections:
            param_content = sections[key]
            break

    returns_content = sections.get("returns")
    required_content = sections.get("required")

    param_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:\([^)]+\))?\s*:\s*(.*)$")
    ret_re = re.compile(r"^\s*([A-Za-z_][\w\[\], ]*)\s*:\s*(.*)$")

    def parse_entries_from(content: str, entry_re: re.Pattern, postprocess_first=None):
        if not content:
            return []
        entries = []
        current_first = None
        current_desc_parts = []
        for raw in content.splitlines():
            if not raw.strip():
                continue
            m = entry_re.match(raw)
            if m:
                if current_first is not None:
                    first = postprocess_first(current_first) if postprocess_first else current_first
                    entries.append((first, collapse(current_desc_parts)))
                current_first = m.group(1).strip()
                first_desc = (m.group(2) or "").strip()
                current_desc_parts = [first_desc] if first_desc else []
            else:
                if current_first is not None:
                    current_desc_parts.append(raw.strip())
        if current_first is not None:
            first = postprocess_first(current_first) if postprocess_first else current_first
            entries.append((first, collapse(current_desc_parts)))
        return entries

    param_entries = parse_entries_from(param_content, param_re)
    returns_entries_raw = parse_entries_from(returns_content, ret_re, postprocess_first=map_doc_type)

    param_descs = {name: desc for name, desc in param_entries}
    returns_entries = [{"type": typ, "description": desc} for typ, desc in returns_entries_raw]

    required_names = []
    if required_content:
        seen = set()
        param_names_order = [name for name, _ in param_entries]
        param_name_set = set(param_names_order)
        for raw in required_content.splitlines():
            for tok in re.findall(r"[A-Za-z_]\w*", raw):
                if tok in param_name_set and tok not in seen:
                    seen.add(tok)
                    required_names.append(tok)

    return func_desc, param_descs, returns_entries, required_names


def extract_name_and_types(func):
    sig = inspect.signature(func)
    annotations = getattr(func, "__annotations__", {}) or {}
    types = {}
    for name, param in sig.parameters.items():
        anno = annotations.get(name, param.annotation)
        types[name] = _map_type_to_schema(anno)
    return func.__name__, types


def get_schema(func):
    func_name, type_map = extract_name_and_types(func)
    func_desc, param_descs, returns_entries, required_names = extract_descriptions(func)

    if not returns_entries:
        ret_anno = getattr(func, "__annotations__", {}).get("return", inspect._empty)
        returns_entries = [{"type": _map_type_to_schema(ret_anno), "description": ""}]

    param_order = list(type_map.keys())
    if not required_names:
        required_names = param_order
    else:
        required_names = [n for n in required_names if n in type_map]

    schema = {
        "name": func_name,
        "description": func_desc,
        "parameters": {},
        "returns": returns_entries,
        "required": required_names,
    }
    for p, t in type_map.items():
        schema["parameters"][p] = {
            "type": t,
            "description": param_descs.get(p, "")
        }
    return schema
