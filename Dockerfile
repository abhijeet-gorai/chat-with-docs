FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
ADD . /app
WORKDIR /app
RUN uv sync --locked
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["uv", "run", "app.py"]