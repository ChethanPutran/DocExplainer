FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies first for better layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy the application and install the project itself
COPY . .
RUN uv sync --frozen

# The backend service (REST /v1, JSON-RPC /rpc, docs at /docs). The desktop UI is not part of this image.
# Server data (uploads, documents, knowledge graphs, chats) lives in /data: mount a volume there to keep it.
ENV DOC_EXPLAINER_DATA_DIR=/data
VOLUME /data
EXPOSE 8000

# Listening on all interfaces requires an API key, so run with:
#   docker run -p 8000:8000 -v doc_explainer_data:/data \
#       -e DOC_EXPLAINER_API_KEY=choose-a-secret -e AIROUTER_API_KEY=your-model-key <image>
CMD ["uv", "run", "doc-explainer-server", "--host", "0.0.0.0", "--port", "8000"]
