# ===============================
# Builder
# ===============================
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY . .

# Install deps into system python
RUN uv sync --system --frozen --no-cache

# ===============================
# Runtime
# ===============================
FROM unclecode/crawl4ai:latest

ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=0  
ENV HOME=/app                   

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# 🔥 MUST RUN IN RUNTIME IMAGE
RUN playwright install chromium

EXPOSE 8000

CMD ["uvicorn", "WebRetrieve_Autonoma.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
