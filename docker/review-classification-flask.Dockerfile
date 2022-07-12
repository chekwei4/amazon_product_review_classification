FROM python:3.8

WORKDIR /app

RUN pip3 install -r requirements.txt

EXPOSE 8080

COPY . /app

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 models/main:app