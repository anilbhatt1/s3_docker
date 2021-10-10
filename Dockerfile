FROM python:3.9.7-slim-buster

MAINTAINER Anil Bhatt

WORKDIR /app

COPY './requirements.txt' .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python"]

CMD ["app.py", "sample_upload_img.jpg"]