# Agent API

This project provides a FastAPI-based server for storing agents and workflows, with token-based access management.

## Features

- Store and manage agent configurations, states, and metadata
- Define and execute workflows that orchestrate agents
- Secure API endpoints with JWT token-based authentication

## Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development)
- Poetry (for local development)

### Using Docker

1. Clone the repository:
   ```bash
   git clone https://github.com/clayxai/EvoAgentX.git
   cd EvoAgentX
   ```

2. Create a `.env` file with your configuration:
   ```bash
      MONGODB_URL=your_mogodb_url
      MONGODB_DB_NAME=evoagentx
   ```

3. Start the services:
   ```bash
   python -m evoagentx.app.main
   ```

4. The API will be available at `http://localhost:8000`
