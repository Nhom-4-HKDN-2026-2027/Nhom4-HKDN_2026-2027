
FROM python:3.11-slim

WORKDIR /app

COPY serving/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY serving/ /app/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
