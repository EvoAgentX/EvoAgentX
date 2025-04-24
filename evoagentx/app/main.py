"""
Main application entry point for EvoAgentX.
"""
import logging
# import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException

from evoagentx.app.config import settings
from evoagentx.app.db import Database
from evoagentx.app.security import init_users_collection
from evoagentx.app.api import (
    auth_router,
    agents_router,
    workflows_router,
    executions_router,
    system_router
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events.
    
    This async context manager handles startup and shutdown events for the FastAPI application.
    On startup, it initializes database connections and creates required collections.
    On shutdown, it ensures all connections are properly closed.
    
    Args:
        app: The FastAPI application instance
        
    Yields:
        None: Control is yielded back to FastAPI after startup is complete
        
    Raises:
        Exception: If startup fails, the exception is logged and re-raised
    """
    # Startup tasks
    try:
        # Connect to database
        await Database.connect()
        
        # Initialize users collection and create admin user if not exists
        await init_users_collection()
        
        logger.info("Application startup completed successfully")
        yield
    except Exception as e:
        logger.error(f"Error during application startup: {e}")
        raise
    finally:
        # Shutdown tasks
        try:
            await Database.disconnect()
            logger.info("Application shutdown completed successfully")
        except Exception as e:
            logger.error(f"Error during application shutdown: {e}")

# Create FastAPI application
app = FastAPI(
    title="EvoAgentX API",
    description="API for EvoAgentX platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(agents_router)
app.include_router(workflows_router)
app.include_router(executions_router)
app.include_router(system_router)

# Global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with a standardized response format.
    
    Captures Pydantic validation errors and returns them in a consistent format
    for API consumers.
    
    Args:
        request: The incoming request that caused the validation error
        exc: The validation error exception
        
    Returns:
        JSONResponse: A structured JSON response with validation error details
    """
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Validation error",
            "errors": exc.errors()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with a standardized response format.
    
    Provides consistent error responses for all HTTP exceptions thrown
    by route handlers.
    
    Args:
        request: The incoming request that caused the exception
        exc: The HTTP exception
        
    Returns:
        JSONResponse: A structured JSON response with error details
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail
        }
    )

# Root endpoint for health check
@app.get("/")
async def root():
    """Provide basic application information and health status.
    
    This endpoint serves as a simple health check to verify that the
    application is running correctly.
    
    Returns:
        dict: Application name, status, and version information
    """
    return {
        "app_name": settings.APP_NAME,
        "status": "healthy",
        "version": "0.1.0"
    }

# Workflow logging and monitoring endpoint
@app.get("/metrics")
async def get_metrics():
    """Retrieve system metrics about agents, workflows, and executions.
    
    Collects and returns statistics about the number of agents, workflows,
    and executions in various states.
    
    Returns:
        dict: Statistics about agents, workflows, and executions
        
    Raises:
        Exception: If metrics collection fails, returns an error message
    """
    # Collect metrics from different services
    try:
        # Collect agent metrics
        total_agents = await Database.agents.count_documents({})
        active_agents = await Database.agents.count_documents({"status": "active"})
        
        # Collect workflow metrics
        total_workflows = await Database.workflows.count_documents({})
        running_workflows = await Database.workflows.count_documents({"status": "running"})
        
        # Collect execution metrics
        total_executions = await Database.executions.count_documents({})
        failed_executions = await Database.executions.count_documents({"status": "failed"})
        
        return {
            "agents": {
                "total": total_agents,
                "active": active_agents
            },
            "workflows": {
                "total": total_workflows,
                "running": running_workflows
            },
            "executions": {
                "total": total_executions,
                "failed": failed_executions
            }
        }
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        return {
            "status": "error",
            "message": "Unable to retrieve metrics"
        }

# Run the application if this script is executed directly
if __name__ == "__main__":
    # Configuration for running the server
    uvicorn_config = {
        "host": settings.HOST,
        "port": settings.PORT,
        "reload": settings.DEBUG,
        "log_level": settings.LOG_LEVEL.lower()
    }
    
    # Start the server
    uvicorn.run("evoagentx.app.main:app", **uvicorn_config)