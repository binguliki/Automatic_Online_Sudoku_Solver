FROM python:3.9-slim-buster

ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_ENABLE_ONEDNN_OPTS=0

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && pip install opencv-python-headless

RUN pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "-b", "0.0.0.0:5000", "wsgi:app"]
