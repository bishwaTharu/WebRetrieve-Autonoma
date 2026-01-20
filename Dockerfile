FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*
# Install uv via pip to avoid ghcr.io credential issues
RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --no-install-project

RUN uv run crawl4ai-setup

COPY . .

# Install the project itself now that the code is present
RUN uv sync

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "WebRetrieve_Autonoma.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
