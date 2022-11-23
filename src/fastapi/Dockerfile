FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /workdir

COPY requirements.txt *.py ./
COPY utils/* ./utils/
RUN apt-get update -y
RUN apt-get install -y python3.10 python3-pip git
RUN pip install -r requirements.txt
EXPOSE 8090
ENTRYPOINT ["python3", "main.py"]
