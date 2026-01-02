FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app


FROM python:3.12.12-alpine AS runtime
LABEL project="resumerating"
LABEL done_by="Akshay"
WORKDIR /Akshay
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app /Akshay
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

