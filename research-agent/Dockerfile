FROM python:3.12-slim

WORKDIR /workspace
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install `uv`
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /uvx /bin/

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-cache

# Copy repo files and sync
COPY . /workspace
RUN uv sync --locked --no-cache

EXPOSE 8888
CMD ["/bin/bash"]