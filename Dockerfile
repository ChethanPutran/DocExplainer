FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies first for better layer caching
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

# Copy application
COPY . .

# Run the application
CMD ["uv", "run", "doc-explainer"]