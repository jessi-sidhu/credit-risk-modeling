FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src/ src/
COPY models/ models/

RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown app:app /app/data
USER app

# SQLite inside the container unless DATABASE_URL is set (docker compose sets Postgres)
ENV DATABASE_URL=sqlite:////app/data/predictions.db
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
