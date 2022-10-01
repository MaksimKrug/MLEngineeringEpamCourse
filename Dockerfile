FROM apache/airflow:2.4.0
USER root
RUN apt update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get install -y ant && \
    apt-get clean;
USER airflow
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r  requirements.txt