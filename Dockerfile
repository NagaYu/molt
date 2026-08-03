# Molt — elastic on-device inference, CPU image.
#
# The image is deliberately CPU-only: the whole point of the project is memory
# pressure on a device, and a CUDA image would invite benchmarking it on a
# machine where the constraint does not exist.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    MOLT_OFFLINE=0

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY molt/ ./molt/
COPY benchmarks/ ./benchmarks/
COPY scripts/ ./scripts/
COPY examples/ ./examples/
COPY README.md pyproject.toml ./

# Weights and fitted projectors are mounted, not baked: they are large, they are
# licensed separately, and a rung's projector must match the exact model it was
# fitted against.
VOLUME ["/models", "/app/artifacts"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Fit projectors first (once), then serve:
#   docker run -v $PWD/artifacts:/app/artifacts -v $HOME/.cache/huggingface:/models molt \
#       python scripts/train_projectors.py --ladder qwen
CMD ["python", "-m", "molt.service", "--ladder", "qwen", "--host", "0.0.0.0", "--port", "8000"]
