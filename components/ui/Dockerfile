FROM python:3.9-slim AS base


FROM base AS build
ARG COMPONENT_NAME

RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*
RUN python -m venv /.venv

COPY components/${COMPONENT_NAME}/requirements.txt /workspace/

WORKDIR /workspace
RUN /.venv/bin/pip install --no-cache-dir -r requirements.txt


FROM base AS release
ARG COMPONENT_NAME

RUN useradd -s /bin/bash appuser

COPY --from=build --chmod=775 /.venv /.venv
COPY components/${COMPONENT_NAME}/app /app

ENV PATH="/.venv/bin:$PATH" \
    PYTHONPATH="/app"

WORKDIR /app
USER appuser

EXPOSE 8050

CMD ["python", "-u", "main.py"]
