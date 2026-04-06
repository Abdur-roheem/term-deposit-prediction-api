FROM python:3.12.1-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY pyproject.toml uv.lock .python-version ./

RUN uv sync --locked

COPY predict_new.py bank-model.bin ./

EXPOSE 9696

ENTRYPOINT ["/code/.venv/bin/uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "9696"]