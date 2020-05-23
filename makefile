build: req container

container:
	docker build -t 'mobile-server' .

req:
	rm -rf requirements.txt
	pip freeze > requirements.txt

restart: stop start

start:
	docker run -d -p 5000:5000 mobile-server > work/id.txt

stop:
	docker kill `cat work/id.txt`

debug:
	docker logs -f `cat work/id.txt`

test:
	curl -F "file=@resources/test.png" http://localhost:5000/calc -o work/test_output.jpg
	echo "Output in work/test_output.jpg"

setup: dependencies build

dependencies: clean_work
	curl https://download.java.net/openjdk/jdk14/ri/openjdk-14+36_linux-x64_bin.tar.gz -o resources/openjdk-14-linux-x64.tar.gz
	tar -zxvf  resources/openjdk-14-linux-x64.tar.gz

clean_work:
	rm -rf work
	mkdir work

clean_all: clean_work
	rm -rf resources/openjdk-14-linux-x64.tar.gz
	rm -rf jdk-14

copy_jar:
	cp ../mobile-conversion/target/mobile-conversion-1.0-SNAPSHOT.jar ./resources/

dupa:
	rm -rf work
