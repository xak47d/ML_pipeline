FROM continuumio/miniconda3:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.7.1

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY main.py config.yaml ./

COPY src/ ./src/

RUN mkdir -p outputs multirun

ENV MLFLOW_CONDA_CREATE_ENV_CMD="false"

CMD ["python", "main.py"]