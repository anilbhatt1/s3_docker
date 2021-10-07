FROM python:3.9.7-slim-buster

MAINTAINER Anil Bhatt

WORKDIR /app

COPY './requirements.txt' .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]