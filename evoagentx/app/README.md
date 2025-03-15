# Project structure
EvoAgentX/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── agents.py
│   │   │   ├── workflows.py
│   │   │   └── auth.py
│   │   └── dependencies.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── security.py
│   │   ├── exceptions.py
│   │   └── logging.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── agent.py
│   │       └── workflow.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── agent_service.py
│   │   ├── workflow_service.py
│   │   └── execution_service.py
│   └── schemas/
│       ├── __init__.py
│       ├── agent.py
│       ├── workflow.py
│       └── auth.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api/
│   │   ├── __init__.py
│   │   ├── test_agents.py
│   │   ├── test_workflows.py
│   │   └── test_auth.py
│   └── test_services/
│       ├── __init__.py
│       ├── test_agent_service.py
│       └── test_workflow_service.py
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md