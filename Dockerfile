FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install mlflow
COPY . .

CMD ["bash", "-c", "python src/prep.py && python src/train.py && python src/eval.py"]