FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory and create reports directory
WORKDIR /app
RUN mkdir -p /app/reports && chmod 777 /app/reports

# Install poetry
RUN pip install --no-cache-dir poetry==1.4.2 && \
    poetry config virtualenvs.create false

# Copy only dependency files first
COPY pyproject.toml poetry.lock ./

# Install dependencies with explicit version constraints
RUN poetry install --no-interaction --no-ansi --no-root \
    && pip install --no-cache-dir "fastapi>=0.104.1" "uvicorn>=0.24.0"

# Copy application code
COPY . .

# Install the package
RUN poetry install --no-interaction --no-ansi

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "forecasting_tools.api.app"]
