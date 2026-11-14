FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared ./shared
COPY services ./services
COPY simulation ./simulation
ARG SERVICE_PATH
ENV PYTHONPATH=/app
CMD ["sh","-c","uvicorn ${SERVICE_PATH}:app --host 0.0.0.0 --port ${PORT}"]
