import os
import time
import psutil
import json
from flask import Flask, request, render_template, redirect, url_for
from txt2img_model import create_pipeline, txt2img, PixArtAlphaTxt2img
from flask_socketio import SocketIO, emit


app = Flask(__name__)
socketio = SocketIO(app)

IMAGE_PATH = "static/output.jpg"

model_list = [        
    "runwayml/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4",
    "prompthero/openjourney",
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    "nota-ai/bk-sdm-small",
    "hakurei/waifu-diffusion",
    "stabilityai/stable-diffusion-2-1",
    "dreamlike-art/dreamlike-photoreal-2.0",    
]

@app.route("/", methods=["GET"])
def index():
    if request.method == "GET":
        return render_template("index.html", model_list=model_list)
    

def get_system_resources():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()

    return {
        'cpu_usage': cpu_usage,
        'memory_used': memory_info.used / 1024**2,  # Convert bytes to MB
        'memory_total': memory_info.total / 1024**2,
    }

@socketio.on('start_txt2img')
def handle_start_txt2img(data):
    prompt = data.get('prompt')
    model = data.get('model')

    start_time = time.time()
    before_resources = get_system_resources()
    if model == "PixArt-alpha/PixArt-XL-2-1024-MS":
        image = PixArtAlphaTxt2img(prompt)
        image.save(IMAGE_PATH)
    else:
        pipeline = create_pipeline(model)
        image = txt2img(prompt, pipeline)    
        image.save(IMAGE_PATH)

    end_time = time.time()
    after_resources = get_system_resources()
    time_taken = end_time - start_time

    emit('image_result', {'image_data': IMAGE_PATH, 'before_resources': json.dumps(before_resources), 'after_resources': json.dumps(after_resources), 'time_taken': time_taken })  # You'll need to handle the image data appropriately


if __name__ == "__main__":    
    socketio.run(app)
