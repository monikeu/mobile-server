FROM ubuntu:18.04
RUN apt-get update && \
	apt-get install -y python3.7 && \
	apt-get install -y python3-pip
COPY resources/openjdk-14-linux-x64.tar.gz /
COPY requirements.txt /
COPY resources/mobile-conversion-1.0-SNAPSHOT.jar /
RUN tar -zxvf  openjdk-14-linux-x64.tar.gz && \
	/usr/bin/python3.7 -m pip install -r requirements.txt && \
	mkdir work

COPY app.py /
COPY model.py /
EXPOSE 5000
ENV FLASK_APP=app.py
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
#CMD which python3 && whereis pythonkubectl logs3 && echo $PATH && flask run --host=0.0.0.0
CMD /usr/bin/python3.7 -m flask run --host=0.0.0.0
