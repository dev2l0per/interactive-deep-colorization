from queue import Empty, Queue
import threading, time, uuid, os
from flask import (
    Flask, send_file, request, Response, render_template,
)

import torch
from data import colorize_image as CI
from skimage import color as skiColor
import numpy as np
from io import BytesIO
from PIL import Image, ImageColor

colorModel = CI.ColorizeImageTorch(Xd=256)

gpu_id = 0 if torch.cuda.is_available() else None

colorModel.prep_net(gpu_id=gpu_id, path='./models/pytorch/caffemodel.pth')

app = Flask(__name__)

requestsQueue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1

def handle_requests_by_batch():
    while True:
        requestsBatch = []
        while not (len(requestsBatch) >= BATCH_SIZE):
            try:
                requestsBatch.append(requestsQueue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for request in requestsBatch:
                request['output'] = run(request['input'][0], request['input'][1], request['input'][2])

threading.Thread(target=handle_requests_by_batch).start()

def put_point(inputAb, mask, loc, p, val):
    inputAb[:, loc[0] - p: loc[0] + p + 1, loc[1] - p : loc[1] + p + 1] = np.array(val)[:, np.newaxis, np.newaxis]
    mask[:, loc[0] - p : loc[0] + p + 1, loc[1] - p : loc[1] + p + 1] = 1
    return (inputAb, mask)

def run(file, labArr, position):
    try:
        filename = uuid.uuid1()
        tempFilePath = f'./input/{filename}.{file.content_type.split("/")[-1]}'

        image = Image.open(file)
        image = image.resize([256, 256])
        image.save(tempFilePath)

        colorModel.load_image(tempFilePath)

        inputAb = np.zeros((2, 256, 256))
        mask = np.zeros((1, 256, 256))

        (inputAb, mask) = put_point(inputAb, mask, position, 3, [labArr[1], labArr[2]])
        colorModel.net_forward(inputAb, mask)
        imgOutFullRes = colorModel.get_img_fullres()
        pilImage = Image.fromarray(np.uint8(imgOutFullRes)).convert('RGB')
        result = BytesIO()
        pilImage.save(result, format=file.content_type.split("/")[-1])
        result.seek(0)
        os.remove(tempFilePath)

        return result
    except Exception as e:
        return "error"

@app.route('/ideepcolor', methods=['POST'])
def ideepcolor():
    if requestsQueue.qsize() > BATCH_SIZE:
        return Response('Too Many Requests', status=429)

    try:
        file = request.files['image']
        color = request.form['color']
        positionX = request.form['positionX']
        positionY = request.form['positionY']
    except:
        return Response("Bad Request", status=400)
    

    rgbArray = ImageColor.getrgb(color)
    rgbNumpyArr = np.array((rgbArray[0], rgbArray[1], rgbArray[2])).astype('uint8')
    labArr = skiColor.rgb2lab(rgbNumpyArr[np.newaxis, np.newaxis, :]).flatten()
    position = [int(positionY), int(positionX)]

    req = {
        'input': [file, labArr, position]
    }

    requestsQueue.put(req)

    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)
    
    io = req['output']
    if io == "error":
        return Response('Server Error', status=500)
    
    return send_file(io, mimetype=f'image/{file.content_type.split("/")[-1]}')

@app.route('/test_imgs/<file_path>')
def get_test_image(file_path: str):
  return send_file('./test_imgs/' + file_path)

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/healthz', methods=['GET'])
def healthz():
    return 'ok'

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")