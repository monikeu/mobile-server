FROM ubuntu:18.04
RUN apt-get update && \
	apt-get install -y python3 && \
	apt-get install -y python3-pip
COPY resources/openjdk-14-linux-x64.tar.gz /
COPY requirements.txt /
COPY resources/mobile-conversion-1.0-SNAPSHOT.jar /
RUN tar -zxvf  openjdk-14-linux-x64.tar.gz && \
	pip3 install -r requirements.txt && \
	mkdir work

COPY app.py /
EXPOSE 5000
ENV FLASK_APP=app.py
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
CMD flask run --host=0.0.0.0
