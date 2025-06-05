FROM python:3.9-bullseye

WORKDIR /app
COPY . .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir \
      streamlit \
      pandas \
      seaborn \
      matplotlib \
      requests \
      scikit-learn \
      tqdm \
      openpyxl \
      xlrd

ENV PYTHONPATH="/app"
CMD ["streamlit", "run", "main.py", "--server.port=8501"]
