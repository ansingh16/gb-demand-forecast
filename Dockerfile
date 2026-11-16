# Live forecast service. Build after `dvc repro` (or `python -m gb_demand.pipeline train`),
# which writes the models this image serves:
#   docker build -t gb-demand .
#   docker run -p 8000:8000 gb-demand
FROM ghcr.io/astral-sh/uv:0.8.22-python3.12-bookworm-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# dependencies first, so code changes do not reinstall them
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY src ./src
COPY params.yaml ./
RUN uv sync --frozen --no-default-groups

COPY models/latest ./models/latest
ENV GB_DEMAND_MODELS=/app/models/latest PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "gb_demand.api:app", "--host", "0.0.0.0", "--port", "8000"]
