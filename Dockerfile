FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip uv

WORKDIR /app
COPY . .

RUN uv sync

RUN uv run crawl4ai-setup
EXPOSE 7860

# Start FastAPI
CMD ["uv", "run", "uvicorn", "WebRetrieve_Autonoma.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
