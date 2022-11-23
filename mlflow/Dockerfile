FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /workdir

RUN apt-get update -y
RUN apt-get install -y python3.10 python3-pip git
RUN pip install mlflow==1.3.0
ENTRYPOINT ["mlflow", "server", "--host",  "0.0.0.0"]