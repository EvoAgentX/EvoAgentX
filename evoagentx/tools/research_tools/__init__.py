# Apply compatibility patches BEFORE importing any sub-modules.
# This ensures that symbols like ToolMetadata, add_dict, track_cost etc.
# are available in the project modules that the research tools import from.
from . import _compat  # noqa: F401  — side-effect-only import

from .toolkit import ResearchToolkit
from .paper_search import ArxivPaperSearchTool

# The following tools have heavier dependencies (crawl4ai, serpapi).
# Guard them so the package is still importable when those deps are absent.
try:
    from .paper_search import PaperSearchTool
except ImportError:
    PaperSearchTool = None  # type: ignore[assignment,misc]

try:
    from .fetch_paper import FetchPaperMetaDataTool
except ImportError:
    FetchPaperMetaDataTool = None  # type: ignore[assignment,misc]

try:
    from .bib_reference import BibReferenceTool
except ImportError:
    BibReferenceTool = None  # type: ignore[assignment,misc]

__all__ = [
    "ResearchToolkit",
    "ArxivPaperSearchTool",
    "PaperSearchTool",
    "FetchPaperMetaDataTool",
    "BibReferenceTool",
]
