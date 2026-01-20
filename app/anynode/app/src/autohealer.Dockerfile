FROM python:3.12-slim
WORKDIR /app
COPY Systems/services/autohealer/src/ /app/
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic httpx
CMD ["uvicorn","auto_healing_trigger:app","--host","0.0.0.0","--port","9034"]
