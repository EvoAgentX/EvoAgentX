import asyncio
from pathlib import Path
from typing import Union, List, Tuple, Optional, Callable, Dict

from llama_index.core import SimpleDirectoryReader

from evoagentx.rag.schema import Document
from evoagentx.core.logging import logger


# You Could fllow the llama_index tutorial to develop a valid Reader for new file format:
# https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/
class LLamaIndexReader:
    """A universal file reader based on LlamaIndex's SimpleDirectoryReader.

    This class provides a flexible interface for loading documents from files or directories,
    supporting various formats (e.g., PDF, Word, Markdown) with customizable filtering and metadata.

    Attributes:
        recursive (bool): Whether to recursively load files from directories.
        exclude_hidden (bool): Whether to exclude hidden files (starting with '.').
        num_workers (Optional[int]): Number of worker threads for parallel loading.
        num_files_limits (Optional[int]): Maximum number of files to load.
        custom_metadata_function (Optional[Callable]): Custom function to extract metadata.
        extern_file_extractor (Optional[Dict]): Custom file extractors for specific file types.
        errors (str): Error handling strategy for file reading (e.g., 'ignore', 'strict').
        encoding (str): File encoding (default: 'utf-8').
    """

    def __init__(
        self,
        recursive: bool = False,
        exclude_hidden: bool = True,
        num_workers: Optional[int] = None,
        num_files_limits: Optional[int] = None,
        custom_metadata_function: Optional[Callable] = None,
        extern_file_extractor: Optional[Dict] = None,
        errors: str = "ignore",
        encoding: str = "utf-8",
    ):
        self.recursive = recursive
        self.exclude_hidden = exclude_hidden
        self.num_workers = num_workers
        self.num_files_limits = num_files_limits
        self.custom_metadata_function = custom_metadata_function
        self.extern_file_extractor = extern_file_extractor
        self.errors = errors
        self.encoding = encoding

    def _validate_path(self, path: Union[str, Path]) -> Path:
        """Validate and convert a path to a Path object.

        Args:
            path: A string or Path object representing a file or directory.

        Returns:
            Path: A validated Path object.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the path is invalid.
        """
        path = Path(path)
        if not path.exists():
            logger.error(f"Path does not exist: {path}")
            raise FileNotFoundError(f"Path does not exist: {path}")
        return path

    def _check_input(
        self, input_data: Union[str, List, Tuple], is_file: bool = True
    ) -> Union[List[Path], Path]:
        """Check input to a list of Path objects or a single Path for directories.

        Args:
            input_data: A string, list, or tuple of file/directory paths.
            is_file: Whether to treat input as file paths (True) or directory (False).

        Returns:
            Union[List[Path], Path]: Valied file paths or directory path.

        Raises:
            ValueError: If input type is invalid.
        """
        if isinstance(input_data, str):
            return self._validate_path(input_data)
        elif isinstance(input_data, (list, tuple)):
            if is_file:
                return [self._validate_path(p) for p in input_data]
            else:
                return self._validate_path(input_data[0])
        else:
            logger.error(f"Invalid input type: {type(input_data)}")
            raise ValueError(f"Invalid input type: {type(input_data)}")

    def load(
        self,
        file_paths: Union[str, List, Tuple],
        exclude_files: Optional[Union[str, List, Tuple]] = None,
        filter_file_by_suffix: Optional[Union[str, List, Tuple]] = None,
        merge_by_file: bool = False,
        show_progress: bool = False,
        use_async: bool = False,
    ) -> List[Document]:
        """Load documents from files or directories.

        Args:
            file_paths: A string, list, or tuple of file paths or a directory path.
            exclude_files: Files to exclude from loading.
            filter_file_by_suffix: File extensions to include (e.g., ['.pdf', '.docx']).

        Returns:
            List[Document]: List of loaded documents.

        Raises:
            FileNotFoundError: If input paths are invalid.
            RuntimeError: If document loading fails.
        """
        try:
            input_files = None
            input_dir = None
            if isinstance(file_paths, (list, tuple)):
                input_files = self._check_input(file_paths, is_file=True)
            else:
                path = self._check_input(file_paths, is_file=False)
                if path.is_dir():
                    input_dir = path
                else:
                    input_files = [path]

            exclude_files = (
                self._check_input(exclude_files, is_file=True)
                if exclude_files
                else None
            )
            filter_file_by_suffix = (
                list(filter_file_by_suffix)
                if isinstance(filter_file_by_suffix, (list, tuple))
                else [filter_file_by_suffix]
                if isinstance(filter_file_by_suffix, str)
                else None
            )

            reader = SimpleDirectoryReader(
                input_dir=input_dir,
                input_files=input_files,
                exclude=exclude_files,
                exclude_hidden=self.exclude_hidden,
                recursive=self.recursive,
                required_exts=filter_file_by_suffix,
                num_files_limit=self.num_files_limits,
                file_metadata=self.custom_metadata_function,
                file_extractor=self.extern_file_extractor,
                encoding=self.encoding,
                errors=self.errors,
            )

            llama_docs = asyncio.run(reader.aload_data(show_progress=show_progress, num_workers=self.num_workers)) if use_async \
                else reader.load_data(show_progress=show_progress)
            
            if merge_by_file:
                file_to_docs = {}
                for doc in llama_docs:
                    file_path = doc.metadata.get("file_path", "")
                    if file_path not in file_to_docs:
                        file_to_docs[file_path] = []
                    file_to_docs[file_path].append(doc)
                
                documents = []
                for file_path, docs in file_to_docs.items():
                    combined_text = "\n".join(doc.text for doc in docs)
                    combined = docs[0].copy()
                    combined.text_resource.text = combined_text
                    combined.metadata["page_count"] = len(docs)
                    documents.append(Document.from_llama_document(combined))
            else:
                documents = [Document.from_llama_document(doc) for doc in llama_docs]
            logger.info(f"Loaded {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"Failed to load documents: {str(e)}")
            raise RuntimeError(f"Failed to load documents: {str(e)}")