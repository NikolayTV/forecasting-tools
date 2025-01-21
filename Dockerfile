FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Set working directory
WORKDIR /app

# Create reports directory with proper permissions
RUN mkdir -p /app/reports && chmod 777 /app/reports

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Install FastAPI and uvicorn explicitly
RUN pip install "fastapi>=0.104.1" "uvicorn>=0.24.0"

# Copy application code
COPY . .

# Install the package
RUN poetry install --no-interaction --no-ansi

# Expose port
EXPOSE 8000

# Command to run the application
