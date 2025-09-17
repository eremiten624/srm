import time

import gunicorn
from flask import Flask, render_template, request, jsonify, Response
import mmap
import posix_ipc
import struct

shm_name = "srm_viewer_web_armor"
shm_size = 0x100000

path = "config.toml"
app = Flask(__name__)

import tomli

with open(path, "rb") as f:
    config = tomli.load(f)


def read_fild(path):
    with open(path, "r") as f:
        data = f.read()
    return data


def write_file(path, data):
    with open(path, "w") as f:
        f.write(data)


def generate():
    shm = posix_ipc.SharedMemory(shm_name)
    shm_map = mmap.mmap(shm.fd, shm_size, mmap.MAP_SHARED, mmap.PROT_READ)
    while True:
        frame_size = struct.unpack("i", shm_map[:4])[0]
        frame_data = shm_map[4:frame_size]
        if frame_data is None:
            continue
        ###卡大概是因为刷新不完？
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n\r\n"
        )
        time.sleep(1 / 40)  # 视频显示在30帧左右即可


@app.route("/")
def index():
    config = read_fild(path)
    return render_template("video.html", config=config)


@app.route("/VideoStream")
def VideoStream():
    return Response(
        generate(), content_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/save", methods=["POST"])
def save():
    data = request.form["content"]
    write_file(path, data)
    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run("0.0.0.0", config["viewer"]["web"]["port"], True, threaded=True)
