FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    streamlit pandas seaborn matplotlib requests scikit-learn tqdm openpyxl xlrd

ENV PYTHONPATH="/app"

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.fileWatcherType", "none"]
