FROM python:3.12-slim
WORKDIR /app
COPY Systems/services/compactifai/src/ /app/
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic httpx
CMD ["uvicorn","compressor:app","--host","0.0.0.0","--port","9031"]
