FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY --from=builder /app/ /app/
COPY services/gateway/app /app/app

EXPOSE 8000
CMD ["poetry", "run", "start"]

# Stage 1: Build stage to install dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy only the files needed for dependency installation
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-root --without dev

# Stage 2: Final stage
FROM python:3.11-slim

WORKDIR /app

# Create a non-root user
RUN useradd -m runner
USER runner

# Copy installed dependencies from the builder stage
COPY --from=builder --chown=runner:runner /app/.venv ./.venv

# Add the virtual environment to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy the application code
COPY --chown=runner:runner app ./app

# Expose the port the app runs on
EXPOSE 1313

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "1313"]
   main

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

     main
WORKDIR /srv/gw
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python","-m","uvicorn","metatron_gate:app","--host","0.0.0.0","--port","8000"]

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv/gw
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python","-m","uvicorn","metatron_gate:app","--host","0.0.0.0","--port","8000"]

     main
