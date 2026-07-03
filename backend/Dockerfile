FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY .venv .venv
ENV PATH=/app/.venv/bin:$PATH
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages -r requirements.txt

COPY . .

RUN mkdir -p uploads log static/result models runs && \
    chmod +x log-wrapper.sh

EXPOSE 8000

ENTRYPOINT ["/app/log-wrapper.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
