FROM python:3.12-slim
WORKDIR /app
RUN pip install fastapi==0.111.0 uvicorn[standard]==0.30.1 httpx==0.27.0
COPY llm_router.py ./
EXPOSE 8009
CMD ["uvicorn","llm_router:app","--host","0.0.0.0","--port","8009"]
