FROM python:3.12-slim

WORKDIR /app

COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Also install requests for the verification script, just in case
RUN pip install --no-cache-dir requests

COPY . .
