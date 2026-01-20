FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir uv

WORKDIR /app
COPY . .


RUN uv sync --frozen --no-cache

# ===============================
# Runtime stage (crawl4ai base)
# ===============================
FROM unclecode/crawl4ai:latest

# IMPORTANT: ensure playwright uses system path
ENV PLAYWRIGHT_BROWSERS_PATH=0
ENV PYTHONUNBUFFERED=1

WORKDIR /app


COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app /app
RUN playwright install chromium

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "WebRetrieve_Autonoma.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
