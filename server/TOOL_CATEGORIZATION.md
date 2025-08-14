# Tool Categorization System

## Overview

The EvoAgentX server now uses a sophisticated tool categorization system that separates tools based on their storage requirements. This ensures that tools are properly configured with storage handlers when needed, while maintaining clean separation of concerns.

## Centralized Server Configuration

The system uses a centralized `SERVER_CONFIG` that provides consistent, server-appropriate defaults for all tools:

```python
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
```

## Tool Information for Generation

The system provides tool information for workflow generation without requiring actual tool instances:

```python
# Get all available tools for generation
from server.utils import get_tools_for_generation
tools_info = get_tools_for_generation()

# Get tools by category
from server.utils import get_tools_by_category
search_tools = get_tools_by_category("search")
database_tools = get_tools_by_category("database")

# Get storage-dependent tools
from server.utils import get_storage_dependent_tools
storage_tools = get_storage_dependent_tools()

# Get API key-dependent tools
from server.utils import get_api_key_dependent_tools
api_tools = get_api_key_dependent_tools()
```

## Tool Categories

### 1. Default Tools (No Storage Handler Required)

These tools operate independently and don't require file storage operations:

- **MCPToolkit** - Model Context Protocol communication
- **WikipediaSearchToolkit** - Wikipedia search operations (with server defaults)
- **GoogleSearchToolkit** - Google search operations (with server defaults)
- **GoogleFreeSearchToolkit** - Free Google search operations (with server defaults)
- **DDGSSearchToolkit** - DuckDuckGo search operations (with server defaults)
- **RSSToolkit** - RSS feed reading
- **RequestToolkit** - HTTP request operations
- **PostgreSQLToolkit** - PostgreSQL database operations (with server defaults)
- **MongoDBToolkit** - MongoDB database operations (with server defaults)
- **FileToolkit** - Basic file operations

### 2. Storage-Dependent Tools (Require Storage Handler)

These tools need a storage handler for file operations:

- **ArxivToolkit** - Downloads and saves PDF papers
- **FaissToolkit** - Vector database persistence and file operations
- **CMDToolkit** - Command execution with file operations
- **StorageToolkit** - Direct storage operations
- **FluxImageGenerationToolkit** - Saves generated images
- **OpenAIImageGenerationToolkit** - Saves generated images
- **ImageAnalysisToolkit** - Reads images and PDFs for analysis

## Tool Information Structure

Each tool in the generation list includes:

```python
{
    "name": "ToolkitName",
    "description": "Human-readable description",
    "capabilities": ["capability1", "capability2"],
    "category": "tool_category",
    "requires_storage": True/False,  # Optional
    "requires_api_key": "API_KEY_NAME"  # Optional
}
```

## Server Defaults Applied

### Search Tools
- **num_search_pages**: 10 (instead of 5) - Reasonable server limit
- **max_content_words**: 2000 (instead of unlimited) - Prevent memory issues
- **max_summary_sentences**: 3 (instead of unlimited) - Concise summaries

### Database Tools
- **auto_save**: True - Always save changes on server
- **read_only**: False - Server needs write access
- **local_path**: None - No local storage on server

### File Operations
- **image_save_dir**: "images" - Appended to storage handler base path
- **temp_working_dir**: "temp" - Appended to storage handler base path
- **analysis_working_dir**: "analysis" - Appended to storage handler base path

## Storage Handler Configuration

The system automatically creates a `SupabaseStorageHandler` using these environment variables:

```bash
SUPABASE_URL_STORAGE="your_supabase_storage_url"
SUPABASE_KEY_STORAGE="your_supabase_storage_key"
SUPABASE_BUCKET_STORAGE="your_storage_bucket_name"
```

The storage handler is configured with a project-specific path:
```
/projects/{project_short_id}/files
```

## Tool Creation Process

1. **Storage Handler Creation**: Creates Supabase storage handler if configuration is available
2. **Default Tools**: Creates all tools that don't need storage with server defaults
3. **Storage Tools**: Creates tools that require storage, injecting the storage handler
4. **Fallback**: If storage configuration is missing, storage tools are disabled but default tools still work

## Benefits

- **Clean Separation**: Tools are properly categorized based on actual requirements
- **Generation Support**: Provides tool information for workflow generation without instances
- **Consistent Defaults**: Server-appropriate defaults for all tools
- **Flexible Configuration**: Storage tools can be enabled/disabled based on configuration
- **Project Isolation**: Each project gets its own storage path
- **Error Handling**: Graceful fallback when storage is unavailable
- **Maintainability**: Centralized configuration makes the system easier to manage
- **Predictable Behavior**: Server tools have consistent, optimized settings

## Usage

### For Workflow Generation
```python
from server.utils import get_tools_for_generation

# Get tool information for planning workflows
tools_info = get_tools_for_generation()
```

### For Tool Execution
```python
from server.utils import create_tools

# Create tools for a specific project
tools = create_tools("project_abc123")

# Create tools with database information
tools = create_tools("project_abc123", database_info)
```

## Excluded Tools

The following tools are intentionally excluded from the server:
- **DockerInterpreterToolkit** - Security concerns
- **BrowserToolkit** - Browser automation not needed on server
- **BrowserUseToolkit** - Browser automation not needed on server
- **PythonInterpreterToolkit** - Security concerns
