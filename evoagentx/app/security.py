"""
Security components for authentication and authorization.
"""
import jwt
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, ValidationError
from pymongo.errors import DuplicateKeyError
from bson import ObjectId

from evoagentx.app.config import settings
from evoagentx.app.db import Database
from evoagentx.app.schemas import TokenPayload, UserCreate, UserResponse

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_PREFIX}/auth/login")

# User model for database
class UserInDB(BaseModel):
    _id: Optional[ObjectId] = None
    email: str
    hashed_password: str
    full_name: Optional[str] = None
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime = datetime.utcnow()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password for storing."""
    return pwd_context.hash(password)

async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get a user by email."""
    return await Database.db.users.find_one({"email": email})

async def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate a user by email and password."""
    user = await get_user_by_email(email)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    if not user.get("is_active", True):
        return None
    return user

async def create_user(user_create: UserCreate) -> UserResponse:
    """Create a new user."""
    # Check if user already exists
    existing_user = await get_user_by_email(user_create.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    user_dict = user_create.dict()
    hashed_password = get_password_hash(user_dict.pop("password"))
    
    new_user = {
        "email": user_dict["email"],
        "hashed_password": hashed_password,
        "full_name": user_dict.get("full_name"),
        "is_active": True,
        "is_admin": False,
        "created_at": datetime.utcnow()
    }
    
    try:
        insert_result = await Database.db.users.insert_one(new_user)
        new_user["_id"] = insert_result.inserted_id
        return UserResponse(**new_user)
    except DuplicateKeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create a new JWT access token."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {"exp": expire, "sub": subject}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Get the current user from a JWT token."""
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
        
        if datetime.fromtimestamp(token_data.exp) < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except (jwt.PyJWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = await get_user_by_email(token_data.sub)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user

async def get_current_active_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Get the current active user."""
    if not current_user.get("is_active", True):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Get the current admin user."""
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

# Initialize the users collection
async def init_users_collection():
    """Initialize the users collection with indexes."""
    await Database.db.users.create_index("email", unique=True)
    
    # Create admin user if it doesn't exist
    admin_email = "admin@clayx.ai"
    admin = await get_user_by_email(admin_email)
    if not admin:
        admin_user = UserCreate(
            email=admin_email,
            password="adminpassword",  # Change this in production!
            full_name="Admin User"
        )
        user_dict = admin_user.dict()
        hashed_password = get_password_hash(user_dict["password"])
        
        new_admin = {
            "email": admin_email,
            "hashed_password": hashed_password,
            "full_name": "Admin User",
            "is_active": True,
            "is_admin": True,
            "created_at": datetime.utcnow()
        }
        
        await Database.db.users.insert_one(new_admin)