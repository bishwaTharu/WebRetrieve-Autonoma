FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache


COPY . .
RUN uv run crawl4ai-setup
RUN uv run playwright install

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "WebRetrieve_Autonoma.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
