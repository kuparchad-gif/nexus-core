FROM python:3.11-slim
ARG MODULE
ENV MODULE_NAME=${MODULE} \
    NEXUS_DIR=/Nexus/${MODULE} \
    OVERLAY_DIR=/Overlay/${MODULE} \
    PACKS_DIR=/Packs/${MODULE} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl jq netcat-openbsd procps iproute2 tini supervisor \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY docker/app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY docker/app/agent.py /app/agent.py
COPY docker/app/panel.py /app/panel.py
COPY docker/app/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY docker/app/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
EXPOSE 8715
HEALTHCHECK --interval=15s --timeout=5s --retries=10 CMD curl -fsS http://127.0.0.1:8715/health || exit 1
ENTRYPOINT ["/usr/bin/tini","--","/usr/local/bin/entrypoint.sh"]
CMD ["sh","-lc","uvicorn panel:app --host 0.0.0.0 --port 8715"]
