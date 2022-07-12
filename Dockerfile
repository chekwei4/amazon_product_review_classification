# FROM python:3.6-slim-buster
# WORKDIR /app
# COPY . .
# RUN pip install -r requirements.txt
# # CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
# CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]

FROM python:3.8-slim-buster

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
