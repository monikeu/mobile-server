import subprocess
import time
import uuid

from flask import Flask, request, send_file, make_response

from model import *

app = Flask(__name__)
global_model = Mobilenet(input_size=8, lr=0.005, momentum=0.9)


@app.route('/calc', methods=['POST'])
def calc():
    data = request.files['file'].read()
    filename = "work/{0}.png".format(str(uuid.uuid1()))
    f = open(filename, "wb+")
    f.write(data)
    f.close()

    time_mills = int(subprocess.check_output(
        ["jdk-14/bin/java", "-jar", "mobile-conversion-1.0-SNAPSHOT.jar", filename]))

    # above command produces below file
    converted = filename.replace(".png", ".jpg")

    resp = make_response(send_file(converted))
    resp.headers['time'] = str(time_mills)

    return resp


@app.route('/updateModel', methods=['POST', 'GET'])
def updateModel():
    request_data = request.json
    global_model.updateModel(request_data['input'], request_data['result'], request_data['mode'])
    print("aahaga")
    return ('', 200)


@app.route('/test')
def test():
    ratio_ = time.time().as_integer_ratio()[0]
    f = open("model/test-file-" + str(ratio_), "w+")
    f.write(str(ratio_))
    return ('', 200)


if __name__ == '__main__':
    app.run()
