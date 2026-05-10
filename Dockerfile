# Base image — slim Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (cached layer — only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY env/       ./env/
COPY agent/     ./agent/
COPY api/       ./api/
COPY models/    ./models/

# Expose FastAPI port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]