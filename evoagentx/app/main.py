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
    """
    Async context manager to handle application startup and shutdown events.
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
    """
    Custom validation error handler to provide more detailed error responses.
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
    """
    Custom HTTP exception handler to provide consistent error responses.
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
    """
    Root endpoint for application health check.
    """
    return {
        "app_name": settings.APP_NAME,
        "status": "healthy",
        "version": "0.1.0"
    }

# Workflow logging and monitoring endpoint
@app.get("/metrics")
async def get_metrics():
    """
    Endpoint to retrieve system metrics and stats.
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