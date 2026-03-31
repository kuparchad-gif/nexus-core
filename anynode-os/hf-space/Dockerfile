FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# HuggingFace Spaces expects port 7860 by default, but we set app_port=8000 in README
ENV ANYNODE_ID=anynode-hf-spaces
ENV ANYNODE_PLATFORM=huggingface
ENV HEALTH_CHECK_INTERVAL=30
ENV PROBE_TIMEOUT=5

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
