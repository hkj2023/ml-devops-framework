FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Copy dependency files
COPY requirements.txt .
COPY wheels/ /wheels/

# Copy project files
COPY . .

# Install dependencies (offline)
RUN pip install --no-index --find-links=/wheels -r requirements.txt
RUN pip install --no-index --find-links=/wheels mlflow-skinny==3.12.0 --no-deps

# Expose Streamlit port
EXPOSE 8501

# Start dashboard
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]