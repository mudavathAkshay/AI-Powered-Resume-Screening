FROM python:3.10-slim AS builder
WORKDIR /app
ADD . /app
RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.12.12-alpine AS runtime
LABEL project="AI-Powered Resume Screening"
LABEL done_by="Akshay"
COPY --from=builder /app /Akshay
WORKDIR /Akshay
CMD ["python", "app.py"]
