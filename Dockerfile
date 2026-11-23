FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install --no-cache-dir -e '.[test]'
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
