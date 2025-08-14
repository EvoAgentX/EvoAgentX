# Use Python 3.12 slim image for better package compatibility
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for selenium and other tools
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    python3-tk \
    tk-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install base requirements first (most stable)
COPY requirements-base.txt ./
RUN pip install --no-cache-dir -r requirements-base.txt

# Copy and install AI/ML requirements
COPY requirements-ai.txt ./
RUN pip install --no-cache-dir -r requirements-ai.txt

# Copy and install web/FastAPI requirements
COPY requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt

# Copy and install tools requirements
COPY requirements-tools.txt ./
RUN pip install --no-cache-dir -r requirements-tools.txt

# Copy server requirements
COPY server/requirements.txt ./server/
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy only essential directories needed by the server
COPY evoagentx/ ./evoagentx/
COPY examples/ ./examples/
COPY docs/ ./docs/
COPY server/ ./server/
COPY README.md ./

# Set environment variables
ENV PYTHONPATH=/app
ENV LOG_LEVEL=warning
ENV SUPPRESS_WARNINGS=true
ENV VERBOSE_STARTUP=false

# Expose port
EXPOSE 8001

# Run the server
CMD ["python", "-m", "server.main"] 