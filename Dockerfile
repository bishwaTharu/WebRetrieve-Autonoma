FROM unclecode/crawl4ai:latest

ENV PYTHONUNBUFFERED=1
ENV HOME=/app

# 🔥 FIX: install browsers into writable path
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV UV_NO_VENV=1

WORKDIR /app

# Ensure browser directory exists and is writable
RUN mkdir -p /ms-playwright && chmod -R 777 /ms-playwright

COPY . .

# Install deps + browsers safely
RUN python -m pip install --no-cache-dir uv \
    && python -m uv sync --frozen --no-cache \
    && python -m playwright install chromium

EXPOSE 8000

CMD ["uvicorn", "WebRetrieve_Autonoma.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
