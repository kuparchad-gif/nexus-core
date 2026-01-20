# docker/services/metatron.Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY Systems/services/metatron/src/ /app/
COPY Systems/services/metatron/config /app/config
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic
ENV METATRON_PORT=9021
CMD ["uvicorn","meta_filter:app","--host","0.0.0.0","--port","9021"]
