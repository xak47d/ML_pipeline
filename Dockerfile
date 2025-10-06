FROM continuumio/miniconda3:latest

WORKDIR /app

RUN pip install --no-cache-dir poetry==1.7.1

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY main.py config.yaml ./

COPY src/ ./src/

CMD ["python", "main.py"]