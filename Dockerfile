FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN uv pip install --system --no-cache \
    fastapi==0.162.4 \
    uvicorn==0.54.0 \
    pydantic==2.14.3 \
    numpy==2.5.4 \
    scipy==1.18.1 \
    pandas==3.1.1 \
    scikit-learn==1.9.1 \
    skops==0.16.0

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app app

COPY --chown=app:app api/ /app/api/

USER 10001:10001

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
