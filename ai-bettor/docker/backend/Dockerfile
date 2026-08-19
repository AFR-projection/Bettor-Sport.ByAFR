FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY backend/ ./backend/
COPY frontend/ ./frontend/

RUN pip install --no-cache-dir \
    fastapi uvicorn psycopg2-binary pydantic requests python-dotenv \
    sqlalchemy alembic numpy pytest

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]