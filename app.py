import subprocess
import uuid

from flask import Flask, request, send_file, make_response

app = Flask(__name__)


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


@app.route('/updateModel')
def updateModel():
    return 'Update model'


if __name__ == '__main__':
    app.run()
