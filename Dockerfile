FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Make main script executable
RUN chmod +x oz_3_6_9_unified_hypervisor.py

# Create production entrypoint
RUN echo '#!/usr/bin/env python3\n\
import asyncio\n\
import sys\n\
from oz_3_6_9_unified_hypervisor import Oz369Hypervisor\n\
\n\
async def main():\n\
    oz = Oz369Hypervisor()\n\
    await oz.boot()\n\
    \n\
    # Keep running\n\
    while oz.is_alive:\n\
        await asyncio.sleep(1)\n\
\n\
if __name__ == "__main__":\n\
    asyncio.run(main())' > main.py

EXPOSE 8080

CMD ["python", "main.py"]
