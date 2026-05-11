FROM python:3.10-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
COPY wheels/ /wheels/

RUN pip install --no-index --find-links=/wheels -r requirements.txt
RUN pip install --no-index --find-links=/wheels mlflow-skinny==3.12.0 --no-deps

COPY . .

CMD ["bash"]
