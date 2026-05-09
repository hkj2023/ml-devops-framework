FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install mlflow
# Copy pre-downloaded wheels
COPY wheels/ /wheels/
RUN pip install /wheels/*

COPY . .

CMD ["bash", "-c", "python src/prep.py && python src/train.py && python src/eval.py"]