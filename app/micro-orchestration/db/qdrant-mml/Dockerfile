FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.111.0 uvicorn[standard]==0.30.0 orjson==3.10.3 httpx==0.27.0
COPY services/qdrant-mml /app
EXPOSE 8007
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8007"]
