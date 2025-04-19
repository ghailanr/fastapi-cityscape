FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY webapp ./webapp

CMD ["streamlit", "run", "webapp/streamlit-app.py", "--server.port=8501", "--server.address=0.0.0.0"]
