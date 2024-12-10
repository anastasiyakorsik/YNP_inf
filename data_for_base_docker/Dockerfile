FROM python:3.10

RUN apt-get update && apt-get install -y docker.io
RUN apt-get install -y ffmpeg
RUN apt-get --yes install libgl1

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

WORKDIR /

ENTRYPOINT ["python"]
