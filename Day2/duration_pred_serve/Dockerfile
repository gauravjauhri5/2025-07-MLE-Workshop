FROM python:3.10-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /uvx /bin/

# copy over project files
ADD pyproject.toml uv.lock README.md src/ /app/

WORKDIR /app/

RUN uv sync --frozen

# copy over the models
ADD models/2022-01.pkl 2022-01.pkl

# set the environment variables
ENV MODEL_PATH 2022-01.pkl
ENV VERSION 2022-01-v01
ENV LOGURU_LEVEL INFO

# we use port 9696
EXPOSE 9696

# run the app on startup
# ENTRYPOINT [ "uv", "run", "python", "duration_pred_serve/serve.py" ]
ENTRYPOINT [ "uv", "run", "gunicorn", "--bind=0.0.0.0:9696", "duration_pred_serve.serve:app" ]