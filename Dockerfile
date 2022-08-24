FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /workdir

COPY . .
RUN apt-get update -y
RUN apt-get install -y python3.10 python3-pip git
RUN pip install -r requirements.txt
