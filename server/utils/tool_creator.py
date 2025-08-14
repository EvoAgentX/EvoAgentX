"""
Tool creation utilities for EvoAgentX server.
Creates and configures tools with proper storage handling.
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Centralized server configuration
SERVER_CONFIG = {
    # Database behavior defaults (server-appropriate)
    "database_defaults": {
        "auto_save": True,  # Always auto-save on server
        "read_only": False,  # Server needs write access
        "local_path": None,  # No local storage on server
    },
    
    # File operation defaults (appended to storage handler base path)
    "file_defaults": {
        "image_save_dir": "images",  # Appended to storage handler base path
        "temp_working_dir": "temp",  # Appended to storage handler base path
        "analysis_working_dir": "analysis",  # Appended to storage handler base path
    }
}

# Default tools list for workflow generation (provides tool information without instances)
# This is used by the workflow generator to understand available capabilities
DEFAULT_TOOLS_FOR_GENERATION = [
    # MCP Tools
    {
        "name": "MCPToolkit",
        "description": "Model Context Protocol communication tools",
        "capabilities": ["external_tool_integration", "protocol_communication"],
        "category": "communication"
    },
    
    # Search Tools
    {
        "name": "WikipediaSearchToolkit", 
        "description": "Search Wikipedia for articles and content",
        "capabilities": ["web_search", "content_retrieval", "knowledge_base"],
        "category": "search"
    },
    {
        "name": "GoogleSearchToolkit",
        "description": "Search Google using Custom Search API",
        "capabilities": ["web_search", "content_retrieval", "real_time_info"],
        "category": "search"
    },
    {
        "name": "GoogleFreeSearchToolkit", 
        "description": "Free Google search operations",
        "capabilities": ["web_search", "content_retrieval", "no_api_key_required"],
        "category": "search"
    },
    {
        "name": "DDGSSearchToolkit",
        "description": "DuckDuckGo search operations",
        "capabilities": ["web_search", "content_retrieval", "privacy_focused"],
        "category": "search"
    },
    
    # Content Tools
    {
        "name": "RSSToolkit",
        "description": "RSS feed reading and parsing",
        "capabilities": ["feed_reading", "content_aggregation", "news_monitoring"],
        "category": "content"
    },
    {
        "name": "RequestToolkit",
        "description": "HTTP request operations",
        "capabilities": ["web_requests", "api_calls", "data_fetching"],
        "category": "communication"
    },
    
    # Database Tools
    {
        "name": "PostgreSQLToolkit",
        "description": "PostgreSQL database operations",
        "capabilities": ["sql_queries", "data_management", "relational_database"],
        "category": "database"
    },
    {
        "name": "MongoDBToolkit",
        "description": "MongoDB document database operations",
        "capabilities": ["document_queries", "data_management", "nosql_database"],
        "category": "database"
    },
    
    # File Tools
    {
        "name": "FileToolkit",
        "description": "Basic file operations",
        "capabilities": ["file_reading", "file_writing", "file_management"],
        "category": "file_operations"
    },
    
    # Storage-Dependent Tools (require storage handler)
    {
        "name": "ArxivToolkit",
        "description": "Download and search arXiv papers",
        "capabilities": ["academic_search", "pdf_download", "research_papers"],
        "category": "research",
        "requires_storage": True
    },
    {
        "name": "FaissToolkit", 
        "description": "Vector database operations and semantic search",
        "capabilities": ["vector_search", "semantic_search", "document_indexing"],
        "category": "ai_search",
        "requires_storage": True
    },
    {
        "name": "CMDToolkit",
        "description": "Command line execution with file operations",
        "capabilities": ["command_execution", "system_operations", "file_operations"],
        "category": "system",
        "requires_storage": True
    },
    {
        "name": "StorageToolkit",
        "description": "Comprehensive storage operations",
        "capabilities": ["file_storage", "file_management", "format_support"],
        "category": "storage",
        "requires_storage": True
    },
    
    # Image Generation Tools (require API keys and storage)
    {
        "name": "FluxImageGenerationToolkit",
        "description": "AI image generation using Flux API",
        "capabilities": ["image_generation", "ai_art", "creative_content"],
        "category": "image_generation",
        "requires_storage": True,
        "requires_api_key": "BFL_API_KEY"
    },
    {
        "name": "OpenAIImageGenerationToolkit",
        "description": "AI image generation using OpenAI DALL-E",
        "capabilities": ["image_generation", "ai_art", "creative_content"],
        "category": "image_generation", 
        "requires_storage": True,
        "requires_api_key": "OPENAI_API_KEY"
    },
    {
        "name": "ImageAnalysisToolkit",
        "description": "AI-powered image and PDF analysis",
        "capabilities": ["image_analysis", "pdf_analysis", "content_extraction"],
        "category": "image_analysis",
        "requires_storage": True,
        "requires_api_key": "OPENROUTER_API_KEY"
    }
]

# Helper function to get tool information for generation
def get_tools_for_generation() -> List[Dict[str, Any]]:
    """
    Get the list of available tools for workflow generation.
    This provides tool information without creating actual instances.
    
    Returns:
        List of tool information dictionaries
    """
    return DEFAULT_TOOLS_FOR_GENERATION.copy()

# Helper function to get tools by category
def get_tools_by_category(category: str) -> List[Dict[str, Any]]:
    """
    Get tools filtered by category.
    
    Args:
        category: Tool category to filter by
        
    Returns:
        List of tools in the specified category
    """
    return [tool for tool in DEFAULT_TOOLS_FOR_GENERATION if tool.get("category") == category]

# Helper function to get tools that require storage
def get_storage_dependent_tools() -> List[Dict[str, Any]]:
    """
    Get tools that require storage handler.
    
    Returns:
        List of storage-dependent tools
    """
    return [tool for tool in DEFAULT_TOOLS_FOR_GENERATION if tool.get("requires_storage", False)]

# Helper function to get tools that require API keys
def get_api_key_dependent_tools() -> List[Dict[str, Any]]:
    """
    Get tools that require API keys.
    
    Returns:
        List of API key-dependent tools
    """
    return [tool for tool in DEFAULT_TOOLS_FOR_GENERATION if tool.get("requires_api_key")]

# Import all available tools for actual tool creation
from evoagentx.tools import (
    # Default tools (no storage needed)
    MCPToolkit,
    WikipediaSearchToolkit,
    GoogleSearchToolkit,
    GoogleFreeSearchToolkit,
    DDGSSearchToolkit,
    ArxivToolkit,  # Actually needs storage for PDF downloads
    FaissToolkit,  # Actually needs storage for vector DB persistence
    PostgreSQLToolkit,
    MongoDBToolkit,
    RSSToolkit,
    RequestToolkit,
    CMDToolkit,  # Actually needs storage for file operations
    FileToolkit,
    
    # Storage-dependent tools
    StorageToolkit,
    FluxImageGenerationToolkit,
    OpenAIImageGenerationToolkit,
    ImageAnalysisToolkit,
    
    # Storage handlers
    SupabaseStorageHandler,
    
    # Excluded tools (Docker and Browser)
    # DockerInterpreterToolkit,
    # BrowserToolkit,
    # BrowserUseToolkit,
    # PythonInterpreterToolkit,
)


def create_tools(project_short_id: str, database_information: Dict[str, Any] = None) -> List[Any]:
    """
    Create tools with proper storage handling for the EvoAgentX server.
    
    Args:
        project_short_id: Project short ID for storage path configuration
        database_information: Database configuration information (optional)
        
    Returns:
        List of configured tools
    """
    tools = []
    
    # Create storage handler for tools that need it
    storage_handler = _create_storage_handler(project_short_id)
    
    # Add default tools (no storage needed)
    tools.extend(_create_default_tools())
    
    # Add storage-dependent tools
    if storage_handler:
        tools.extend(_create_storage_tools(storage_handler))
    else:
        print("⚠️  Storage tools disabled due to missing configuration")
    
    print(f"🔧 Created {len(tools)} tools for project {project_short_id}")
    return tools


def _create_storage_handler(project_short_id: str) -> SupabaseStorageHandler:
    """
    Create a Supabase storage handler for the project.
    
    Args:
        project_short_id: Project short ID
        
    Returns:
        Configured SupabaseStorageHandler or None if configuration missing
    """
    supabase_url = os.getenv("SUPABASE_URL_STORAGE")
    supabase_key = os.getenv("SUPABASE_KEY_STORAGE")
    supabase_bucket = os.getenv("SUPABASE_BUCKET_STORAGE")
    
    if not all([supabase_url, supabase_key, supabase_bucket]):
        print("⚠️  Missing Supabase storage configuration:")
        print(f"   SUPABASE_URL_STORAGE: {'✓' if supabase_url else '✗'}")
        print(f"   SUPABASE_KEY_STORAGE: {'✓' if supabase_key else '✗'}")
        print(f"   SUPABASE_BUCKET_STORAGE: {'✓' if supabase_bucket else '✗'}")
        return None
    
    try:
        storage_handler = SupabaseStorageHandler(
            bucket_name=supabase_bucket,
            base_path=f"/projects/{project_short_id}/files"
        )
        print(f"✅ Storage handler created for project {project_short_id}")
        return storage_handler
    except Exception as e:
        print(f"❌ Failed to create storage handler: {e}")
        return None


def _create_default_tools() -> List[Any]:
    """
    Create default tools that don't require storage.
    
    Returns:
        List of default tools
    """
    default_tools = []
    
    try:
        # MCP Toolkit
        mcp_toolkit = MCPToolkit()
        default_tools.append(mcp_toolkit)
        print("✅ MCPToolkit created")
    except Exception as e:
        print(f"❌ Failed to create MCPToolkit: {e}")
    
    try:
        # Search tools with reasonable server defaults
        wiki_toolkit = WikipediaSearchToolkit(
            num_search_pages=10,  # Reasonable limit for server
            max_content_words=2000,  # Prevent memory issues
            max_summary_sentences=3  # Concise summaries
        )
        default_tools.append(wiki_toolkit)
        print("✅ WikipediaSearchToolkit created with server defaults")
    except Exception as e:
        print(f"❌ Failed to create WikipediaSearchToolkit: {e}")
    
    try:
        google_toolkit = GoogleSearchToolkit(
            num_search_pages=10,  # Reasonable limit for server
            max_content_words=2000  # Prevent memory issues
        )
        default_tools.append(google_toolkit)
        print("✅ GoogleSearchToolkit created with server defaults")
    except Exception as e:
        print(f"❌ Failed to create GoogleSearchToolkit: {e}")
    
    try:
        google_free_toolkit = GoogleFreeSearchToolkit(
            num_search_pages=10,  # Reasonable limit for server
            max_content_words=2000  # Prevent memory issues
        )
        default_tools.append(google_free_toolkit)
        print("✅ GoogleFreeSearchToolkit created with server defaults")
    except Exception as e:
        print(f"❌ Failed to create GoogleFreeSearchToolkit: {e}")
    
    try:
        ddgs_toolkit = DDGSSearchToolkit(
            num_search_pages=10,  # Reasonable limit for server
            max_content_words=2000  # Prevent memory issues
        )
        default_tools.append(ddgs_toolkit)
        print("✅ DDGSSearchToolkit created with server defaults")
    except Exception as e:
        print(f"❌ Failed to create DDGSSearchToolkit: {e}")
    
    try:
        # RSS tools
        rss_toolkit = RSSToolkit()
        default_tools.append(rss_toolkit)
        print("✅ RSSToolkit created")
    except Exception as e:
        print(f"❌ Failed to create RSSToolkit: {e}")
    
    try:
        # Request tools
        request_toolkit = RequestToolkit()
        default_tools.append(request_toolkit)
        print("✅ RequestToolkit created")
    except Exception as e:
        print(f"❌ Failed to create RequestToolkit: {e}")
    
    try:
        # Database tools with server defaults
        postgres_toolkit = PostgreSQLToolkit(
            auto_save=SERVER_CONFIG["database_defaults"]["auto_save"],
            local_path=SERVER_CONFIG["database_defaults"]["local_path"]
        )
        default_tools.append(postgres_toolkit)
        print("✅ PostgreSQLToolkit created with server defaults")
    except Exception as e:
        print(f"❌ Failed to create PostgreSQLToolkit: {e}")
    
    try:
        mongo_toolkit = MongoDBToolkit(
            auto_save=SERVER_CONFIG["database_defaults"]["auto_save"],
            local_path=SERVER_CONFIG["database_defaults"]["local_path"],
            read_only=SERVER_CONFIG["database_defaults"]["read_only"]
        )
        default_tools.append(mongo_toolkit)
        print("✅ MongoDBToolkit created with server defaults")
    except Exception as e:
        print(f"❌ Failed to create MongoDBToolkit: {e}")
    
    try:
        # File tools
        file_toolkit = FileToolkit()
        default_tools.append(file_toolkit)
        print("✅ FileToolkit created")
    except Exception as e:
        print(f"❌ Failed to create FileToolkit: {e}")
    
    return default_tools


def _create_storage_tools(storage_handler: SupabaseStorageHandler) -> List[Any]:
    """
    Create tools that require storage handling.
    
    Args:
        storage_handler: Configured storage handler
        
    Returns:
        List of storage-dependent tools
    """
    storage_tools = []
    
    try:
        # Storage toolkit
        storage_toolkit = StorageToolkit(storage_handler=storage_handler)
        storage_tools.append(storage_toolkit)
        print("✅ StorageToolkit created with storage handler")
    except Exception as e:
        print(f"❌ Failed to create StorageToolkit: {e}")
    
    try:
        # ArXiv toolkit (needs storage for PDF downloads)
        arxiv_toolkit = ArxivToolkit(storage_handler=storage_handler)
        storage_tools.append(arxiv_toolkit)
        print("✅ ArxivToolkit created with storage handler")
    except Exception as e:
        print(f"❌ Failed to create ArxivToolkit: {e}")
    
    try:
        # FAISS toolkit (needs storage for vector DB persistence)
        faiss_toolkit = FaissToolkit(
            storage_handler=storage_handler,
            file_handler=storage_handler
        )
        storage_tools.append(faiss_toolkit)
        print("✅ FaissToolkit created with storage handler")
    except Exception as e:
        print(f"❌ Failed to create FaissToolkit: {e}")
    
    try:
        # CMD toolkit (needs storage for file operations)
        cmd_toolkit = CMDToolkit(storage_handler=storage_handler)
        storage_tools.append(cmd_toolkit)
        print("✅ CMDToolkit created with storage handler")
    except Exception as e:
        print(f"❌ Failed to create CMDToolkit: {e}")
    
    try:
        # Flux image generation
        flux_api_key = os.getenv("BFL_API_KEY")
        if flux_api_key:
            flux_toolkit = FluxImageGenerationToolkit(
                api_key=flux_api_key,
                save_path=SERVER_CONFIG["file_defaults"]["image_save_dir"],  # Appended to storage handler base path
                storage_handler=storage_handler
            )
            storage_tools.append(flux_toolkit)
            print("✅ FluxImageGenerationToolkit created with storage handler")
        else:
            print("⚠️  FluxImageGenerationToolkit skipped - missing BFL_API_KEY")
    except Exception as e:
        print(f"❌ Failed to create FluxImageGenerationToolkit: {e}")
    
    try:
        # OpenAI image generation
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_org_id = os.getenv("OPENAI_ORGANIZATION_ID")
        if openai_api_key and openai_org_id:
            openai_img_toolkit = OpenAIImageGenerationToolkit(
                api_key=openai_api_key,
                organization_id=openai_org_id,
                save_path=SERVER_CONFIG["file_defaults"]["image_save_dir"],  # Appended to storage handler base path
                storage_handler=storage_handler
            )
            storage_tools.append(openai_img_toolkit)
            print("✅ OpenAIImageGenerationToolkit created with storage handler")
        else:
            print("⚠️  OpenAIImageGenerationToolkit skipped - missing OpenAI credentials")
    except Exception as e:
        print(f"❌ Failed to create OpenAIImageGenerationToolkit: {e}")
    
    try:
        # Image analysis toolkit
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_api_key:
            image_analysis_toolkit = ImageAnalysisToolkit(
                api_key=openrouter_api_key,
                storage_handler=storage_handler
            )
            storage_tools.append(image_analysis_toolkit)
            print("✅ ImageAnalysisToolkit created with storage handler")
        else:
            print("⚠️  ImageAnalysisToolkit skipped - missing OPENROUTER_API_KEY")
    except Exception as e:
        print(f"❌ Failed to create ImageAnalysisToolkit: {e}")
    
    return storage_tools

