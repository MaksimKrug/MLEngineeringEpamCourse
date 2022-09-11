FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /workdir

COPY requirements.txt *.py ./
RUN apt-get update -y
RUN apt-get install -y python3.10 python3-pip git
RUN pip install -r requirements.txt
RUN python3 get_data.py && python3 preprocess_data.py
