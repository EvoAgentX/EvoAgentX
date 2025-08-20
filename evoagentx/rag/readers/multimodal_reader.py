import asyncio
import io
from pathlib import Path
from typing import Union, List, Tuple, Optional

from pdf2image import convert_from_path
from PIL import Image
from llama_index.core.schema import ImageDocument

from evoagentx.core.logging import logger


class MultimodalReader:
    """A multimodal file reader for images and PDFs.

    This class provides interface for loading images from files or directories,
    supporting various image formats and PDF conversion with pdf2image.

    Attributes:
        recursive (bool): Whether to recursively read directories.
        exclude_hidden (bool): Whether to exclude hidden files (starting with '.').
        num_files_limits (Optional[int]): Maximum number of files to read.
        errors (str): Error handling strategy for file reading (e.g., 'ignore', 'strict').
    """

    def __init__(
        self,
        recursive: bool = False,
        exclude_hidden: bool = True,
        num_files_limits: Optional[int] = None,
        errors: str = "ignore",
    ):
        self.recursive = recursive
        self.exclude_hidden = exclude_hidden
        self.num_files_limits = num_files_limits
        self.errors = errors

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
            Union[List[Path], Path]: Valid file paths or directory path.

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
    ) -> List[ImageDocument]:
        """Load images from files or directories.

        Args:
            file_paths: A string, list, or tuple of file paths or a directory path.
            exclude_files: Files to exclude from loading.
            filter_file_by_suffix: File extensions to include (e.g., ['.png', '.jpg', '.pdf']).
            merge_by_file: Whether to merge documents by file (unused for images, kept for compatibility).

        Returns:
            List[ImageDocument]: List of loaded ImageDocuments.

        Raises:
            FileNotFoundError: If input paths are invalid.
            RuntimeError: If image loading fails.
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

            # Get all files to process
            all_files = []
            if input_files:
                all_files = input_files
            elif input_dir:
                pattern = "**/*" if self.recursive else "*"
                all_files = [f for f in input_dir.glob(pattern) if f.is_file()]
                
                if self.exclude_hidden:
                    all_files = [f for f in all_files if not f.name.startswith('.')]

            # Apply exclusions
            if exclude_files:
                exclude_names = {f.name for f in exclude_files}
                all_files = [f for f in all_files if f.name not in exclude_names]

            # Apply suffix filter
            if filter_file_by_suffix:
                all_files = [f for f in all_files if f.suffix.lower() in filter_file_by_suffix]

            # Apply file limit
            if self.num_files_limits:
                all_files = all_files[:self.num_files_limits]

            # Process files
            documents = []
            for file_path in all_files:
                if show_progress:
                    logger.info(f"Processing: {file_path.name}")
                
                try:
                    if file_path.suffix.lower() == '.pdf':
                        # Convert PDF to images
                        pdf_docs = self._process_pdf(file_path)
                        documents.extend(pdf_docs)
                    elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']:
                        # Process image file
                        img_doc = self._process_image(file_path)
                        if img_doc:
                            documents.append(img_doc)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {str(e)}")
                    if self.errors == "strict":
                        raise

            logger.info(f"Loaded {len(documents)} image documents")
            return documents

        except Exception as e:
            logger.error(f"Failed to load documents: {str(e)}")
            raise RuntimeError(f"Failed to load documents: {str(e)}")

    def _process_image(self, file_path: Path) -> ImageDocument:
        """Process a single image file."""
        try:
            with Image.open(file_path) as img:
                # Convert to RGB for consistency
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                pil_image = img.copy()
                width, height = img.size
                format_name = img.format or 'Unknown'
                
                # Convert PIL Image to bytes for ImageDocument
                img_bytes = io.BytesIO()
                pil_image.save(img_bytes, format='PNG')
                img_bytes = img_bytes.getvalue()

            # Create ImageDocument with same metadata as LlamaIndex SimpleDirectoryReader
            document = ImageDocument(
                text="",
                image=img_bytes,
                image_path=str(file_path),
                image_mimetype=f"image/{format_name.lower()}",
                metadata={
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_type": file_path.suffix,
                    "file_size": file_path.stat().st_size,
                    "creation_date": str(file_path.stat().st_ctime),
                    "last_modified_date": str(file_path.stat().st_mtime)
                }
            )
            
            return document
        except Exception as e:
            logger.error(f"Failed to process image {file_path}: {str(e)}")
            if self.errors == "strict":
                raise
            return None

    def _process_pdf(self, file_path: Path) -> List[ImageDocument]:
        """Convert PDF pages to ImageDocuments."""
        documents = []
        try:
            images = convert_from_path(file_path, dpi=200)
            
            for page_num, img in enumerate(images, 1):
                try:
                    width, height = img.size
                    
                    # Convert PIL Image to bytes for ImageDocument
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='PNG')
                    img_bytes = img_bytes.getvalue()
                    
                    # Create ImageDocument with same metadata as LlamaIndex SimpleDirectoryReader
                    document = ImageDocument(
                        text="",
                        image=img_bytes,
                        image_path=str(file_path),
                        image_mimetype="image/png",
                        metadata={
                            "file_path": str(file_path),
                            "file_name": f"{file_path.stem}_page_{page_num}.png",
                            "file_type": file_path.suffix,
                            "file_size": file_path.stat().st_size,
                            "creation_date": str(file_path.stat().st_ctime),
                            "last_modified_date": str(file_path.stat().st_mtime),
                            "page_label": str(page_num)
                        }
                    )
                    documents.append(document)
                    
                except Exception as e:
                    logger.warning(f"Failed to process page {page_num} of {file_path}: {str(e)}")
                    continue
            
            logger.info(f"Converted {len(documents)} pages from PDF: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to convert PDF {file_path}: {str(e)}")
            if self.errors == "strict":
                raise
                
        return documents