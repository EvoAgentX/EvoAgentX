"""
CORS Configuration for the EvoAgentX Server
"""

# List of allowed origins for CORS
ALLOWED_ORIGINS = ["*"]

# CORS Configuration
CORS_CONFIG = {
    "allow_origins": ALLOWED_ORIGINS,
    "allow_credentials": True,
    "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["*"]
}

def get_cors_config():
    """Get the CORS configuration"""
    return CORS_CONFIG

def add_allowed_origin(origin: str):
    """Add a new allowed origin"""
    if origin not in ALLOWED_ORIGINS:
        ALLOWED_ORIGINS.append(origin)
        CORS_CONFIG["allow_origins"] = ALLOWED_ORIGINS

def remove_allowed_origin(origin: str):
    """Remove an allowed origin"""
    if origin in ALLOWED_ORIGINS:
        ALLOWED_ORIGINS.remove(origin)
        CORS_CONFIG["allow_origins"] = ALLOWED_ORIGINS

def get_allowed_origins():
    """Get the list of allowed origins"""
    return ALLOWED_ORIGINS.copy() 