# 1. Use the official all-in-one image (includes Playwright, browsers, and system deps)
FROM unclecode/crawl4ai:all

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

USER root
RUN pip install --no-cache-dir --upgrade uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache
COPY . .
RUN uv run crawl4ai-doctor
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "WebRetrieve_Autonoma.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
