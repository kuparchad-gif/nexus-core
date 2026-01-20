FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Use PORT from env (each service gets its own)
CMD [ "bash","-lc","uvicorn app:app --host 0.0.0.0 --port \" ]
