# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Metadata
LABEL maintainer="Ankit Umesh Naik <ankitnaik949@gmail.com>"
LABEL description="Customer Support OpenEnv — Meta Hackathon 2025"

# Hugging Face Spaces requires running as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY --chown=user . /app

# Add /app to PYTHONPATH so env/ and baseline/ are importable
ENV PYTHONPATH=/app

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
