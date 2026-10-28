FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY iteration_1/turn_detector_model/ ./iteration_1/turn_detector_model/
COPY iteration_1/main.py ./iteration_1/main.py

EXPOSE 8000

ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

CMD ["uvicorn", "iteration_1.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]