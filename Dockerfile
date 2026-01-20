FROM unclecode/crawl4ai:latest

ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=0
ENV HOME=/app
ENV UV_NO_VENV=1

WORKDIR /app

COPY . .


RUN python -m pip install --no-cache-dir uv \
    && python -m uv sync --frozen --no-cache \
    && playwright install chromium

EXPOSE 8000

CMD ["uvicorn", "WebRetrieve_Autonoma.api.main:app", "--host", "0.0.0.0", "--port", "8000"]