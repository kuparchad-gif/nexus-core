# Spirallaspan.Dockerfile
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Spirallaspan and Memory Substrate scripts
COPY spirallaspan_memory.py .
COPY memory_substrate_protocol.py .

# Expose the API port
EXPOSE 8080

# Run the application in cloud mode
CMD ["python3", "spirallaspan_memory.py"]
