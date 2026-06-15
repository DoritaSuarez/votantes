FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py pipeline.py ./
COPY data/ ./data/
COPY assets/ ./assets/

EXPOSE 8050

CMD ["gunicorn", "app:server", \
     "--bind", "0.0.0.0:8050", \
     "--workers", "1", \
     "--timeout", "120"]
