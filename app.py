import time
import uuid

from flask import Flask, request, make_response, send_file

from model import *

app = Flask(__name__)
global_model = Mobilenet(input_size=1)
from PIL import Image


@app.route('/calc', methods=['POST'])
def calc():
    data = request.files['file'].read()
    filename = f"/home/robert/PycharmProjects/mobile-server/work/{uuid.uuid1()}.png"

    f = open(filename, "wb+")
    f.write(data)
    f.close()

    start = int(time.time() * 1000)
    for i in range(10):
        im = Image.open(filename)
        rgb_im = im.convert('RGB')
        rgb_im.save(f'{filename}.jpg')

    # above command produces below file
    # converted = filename.replace(".png", ".jpg")

    resp = make_response()
    resp.headers['time'] = str(int(time.time() * 1000) - start)

    return resp


@app.route('/fetchModel', methods=['GET'])
def fetchModel():
    resp = make_response(send_file(MODEL_TORCHSCRIPT_PATH))
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
