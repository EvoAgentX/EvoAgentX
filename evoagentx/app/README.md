# EvoAgentX Server

EvoAgentX is a powerful REST API server for managing AI agents, workflows, and their executions. Built with FastAPI, it provides a robust platform for creating, managing, and executing complex AI-powered workflows with support for multiple LLM providers and database backends.

## 🚀 Features

### Core Functionality
- **Agent Management**: Create, update, delete, and query AI agents with different LLM configurations
- **Workflow Creation**: Design complex workflows with multiple steps and agent interactions
- **Workflow Execution**: Execute workflows with real-time monitoring and logging
- **AI-Powered Workflow Generation**: Automatically generate workflows from natural language goals
- **Backup & Restore**: Comprehensive backup and restore functionality for agents and workflows

### Technical Features
- **Multiple Database Support**: MongoDB, PostgreSQL, and in-memory database options
- **JWT Authentication**: Secure user authentication and authorization
- **RESTful API**: Comprehensive REST API with OpenAPI/Swagger documentation
- **Async Operations**: Full async support for high-performance operations
- **Real-time Monitoring**: Execution logging and status tracking
- **CORS Support**: Configurable Cross-Origin Resource Sharing
- **Health Monitoring**: Built-in health checks and metrics endpoints

### LLM Support
- **OpenAI Integration**: Support for GPT models
- **LiteLLM**: Multi-provider LLM support
- **SiliconFlow**: Alternative LLM provider support
- **Flexible Configuration**: Customizable LLM parameters and settings

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL (optional, for PostgreSQL backend)
- MongoDB (optional, for MongoDB backend)
- OpenAI API key (for AI features)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd EvoAgentX
   ```

2. **Install dependencies**:
   ```bash
   pip install -r evoagentx/app/requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp evoagentx/app/sample_app.env evoagentx/app/app.env
   ```

4. **Edit configuration** (see Configuration section below)

## ⚙️ Configuration

### Environment Variables

Create an `app.env` file in the `evoagentx/app/` directory with the following settings:

```env
# Application Settings
APP_NAME=EvoAgentX
DEBUG=True
API_PREFIX=/api/v1
HOST=0.0.0.0
PORT=8000

# Database Settings (choose one)
# MongoDB
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=evoagentx

# PostgreSQL
POSTGRESQL_URL=postgresql://username:password@localhost:5432/dbname

# Supabase (alternative PostgreSQL)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-key

# Authentication
SECRET_KEY=your-super-secret-key-change-this-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
ALGORITHM=HS256

# Logging
LOG_LEVEL=INFO

# CORS Settings
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8000"]
```

### Database Configuration

The server supports multiple database backends:

1. **PostgreSQL** (Recommended): Full-featured relational database
2. **MongoDB**: NoSQL document database
3. **In-Memory**: For testing and development

The database type is automatically detected based on the configuration provided.

## 🚀 Running the Server

### Development Mode
```bash
cd EvoAgentX
python -m evoagentx.app.main
```

### Production Mode
```bash
uvicorn evoagentx.app.main:app --host 0.0.0.0 --port 8000
```

The server will start and be available at:
- API: `http://localhost:8000`
- Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/api/v1/health`

## 📚 API Documentation

### Authentication

#### Register User
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword",
  "full_name": "User Name"
}
```

#### Login
```http
POST /api/v1/auth/login
Content-Type: application/x-www-form-urlencoded

username=user@example.com&password=securepassword
```

Returns a JWT token for authenticated requests.

### Agent Management

#### Create Agent
```http
POST /api/v1/agents
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "My AI Agent",
  "description": "An intelligent agent for specific tasks",
  "config": {
    "model_name": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "tags": ["ai", "assistant"]
}
```

#### Get Agent
```http
GET /api/v1/agents/{agent_id}
Authorization: Bearer <token>
```

#### Update Agent
```http
PUT /api/v1/agents/{agent_id}
Authorization: Bearer <token>
Content-Type: application/json

{
  "description": "Updated description",
  "config": {
    "temperature": 0.5
  }
}
```

#### List Agents
```http
GET /api/v1/agents?skip=0&limit=10&query=search_term
Authorization: Bearer <token>
```

#### Query Agent
```http
POST /api/v1/agents/{agent_id}/query
Authorization: Bearer <token>
Content-Type: application/json

{
  "prompt": "Hello, how can you help me?",
  "history": []
}
```

#### Delete Agent
```http
DELETE /api/v1/agents/{agent_id}
Authorization: Bearer <token>
```

### Workflow Management

#### Create Workflow
```http
POST /api/v1/workflows
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Data Processing Workflow",
  "description": "A workflow for processing data",
  "definition": {
    "steps": [
      {
        "step_id": "step1",
        "agent_id": "agent_id_here",
        "action": "process_data",
        "input_mapping": {"input": "data"},
        "output_mapping": {"output": "result"}
      }
    ]
  },
  "tags": ["data", "processing"]
}
```

#### Execute Workflow
```http
POST /api/v1/executions
Authorization: Bearer <token>
Content-Type: application/json

{
  "workflow_id": "workflow_id_here",
  "input_params": {
    "data": "input data for processing"
  }
}
```

#### Get Execution Status
```http
GET /api/v1/executions/{execution_id}
Authorization: Bearer <token>
```

#### Get Execution Logs
```http
GET /api/v1/executions/{execution_id}/logs
Authorization: Bearer <token>
```

### AI-Powered Workflow Generation

#### Generate Workflow from Goal
```http
POST /api/v1/workflows/generate
Authorization: Bearer <token>
Content-Type: application/json

{
  "goal": "Create a workflow that processes customer feedback and generates insights",
  "llm_config": {
    "model": "gpt-4",
    "temperature": 0.7
  },
  "additional_info": {
    "domain": "customer_service",
    "data_sources": ["emails", "surveys"]
  }
}
```

#### Execute Dynamic Workflow
```http
POST /api/v1/workflows/execute
Authorization: Bearer <token>
Content-Type: application/json

{
  "workflow_graph": {
    "nodes": [...],
    "edges": [...]
  },
  "llm_config": {
    "model": "gpt-3.5-turbo"
  },
  "inputs": {
    "user_input": "process this data"
  }
}
```

### Backup and Restore

#### Backup Agent
```http
POST /api/v1/agents/{agent_id}/backup
Authorization: Bearer <token>
Content-Type: application/json

{
  "backup_path": "/path/to/backup/agent_backup.json"
}
```

#### Restore Agent
```http
POST /api/v1/agents/restore
Authorization: Bearer <token>
Content-Type: application/json

{
  "backup_path": "/path/to/backup/agent_backup.json"
}
```

#### Backup All Agents
```http
POST /api/v1/agents/backup-all
Authorization: Bearer <token>
Content-Type: application/json

{
  "backup_dir": "/path/to/backup/directory"
}
```

## 🏗️ Architecture

### Database Layer
- **Abstract Database Interface**: Unified interface for different database backends
- **MongoDB Implementation**: Full-featured NoSQL implementation
- **PostgreSQL Implementation**: Relational database with JSONB support
- **In-Memory Implementation**: For testing and development

### Service Layer
- **AgentService**: Business logic for agent management
- **WorkflowService**: Workflow creation and management
- **WorkflowExecutionService**: Execution orchestration and monitoring
- **AgentBackupService**: Backup and restore operations
- **WorkflowGeneratorService**: AI-powered workflow generation

### API Layer
- **Authentication Routes**: User registration and login
- **Agent Routes**: CRUD operations for agents
- **Workflow Routes**: Workflow management endpoints
- **Execution Routes**: Workflow execution and monitoring
- **System Routes**: Health checks and metrics
- **Workflow Generator Routes**: AI-powered workflow creation

### Security
- **JWT Authentication**: Secure token-based authentication
- **Role-based Access**: User and admin roles
- **Password Hashing**: Bcrypt password hashing
- **CORS Protection**: Configurable cross-origin policies

## 🔧 Development

### Project Structure
```
evoagentx/app/
├── main.py              # Application entry point
├── config.py            # Configuration management
├── api.py               # API routes and endpoints
├── services.py          # Business logic services
├── db.py                # Database implementations
├── schemas.py           # Pydantic models
├── security.py          # Authentication and security
├── backbone.py          # Workflow generation logic
├── prompts.py           # LLM prompts
├── requirements.txt     # Dependencies
├── app.env              # Environment configuration
└── README.md           # This file
```

### Key Components

#### Database Models
- **Agent**: AI agent configuration and metadata
- **Workflow**: Workflow definition and steps
- **WorkflowExecution**: Execution state and results
- **ExecutionLog**: Detailed execution logging
- **User**: User authentication and authorization

#### Status Enums
- **AgentStatus**: `CREATED`, `ACTIVE`, `INACTIVE`, `ERROR`
- **WorkflowStatus**: `CREATED`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`
- **ExecutionStatus**: `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `TIMEOUT`, `CANCELLED`

### Adding Custom LLM Providers

To add support for new LLM providers:

1. Create a new model class in the `evoagentx.models` package
2. Implement the required interface methods
3. Update the LLM configuration in `backbone.py`
4. Add provider-specific configuration options

### Database Extensions

To add support for new database backends:

1. Implement the `Database` abstract interface in `db.py`
2. Add the new implementation to the `create_database` factory function
3. Update configuration to support the new database type

## 📊 Monitoring and Metrics

### Health Checks
- **Basic Health**: `GET /api/v1/health`
- **Database Health**: Included in health check response
- **System Metrics**: `GET /metrics` (admin only)

### Logging
- Structured logging with configurable levels
- Execution tracking and audit trails
- Error monitoring and debugging information

## 🔒 Security Considerations

- Change the default `SECRET_KEY` in production
- Use strong passwords for database connections
- Enable HTTPS in production deployments
- Regularly update dependencies
- Monitor access logs and authentication attempts
- Use environment variables for sensitive configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review the logs for error messages
3. Ensure proper configuration
4. Verify database connectivity

---

**EvoAgentX** - Empowering AI workflow automation and management.