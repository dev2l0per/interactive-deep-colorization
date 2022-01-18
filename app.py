from flask import (
    Flask, send_file, request, Response, render_template,
)

from data import colorize_image as CI
from skimage import color as skiColor
import numpy as np
import cv2 as cv
from io import BytesIO
from PIL import Image, ImageColor
import os

import uuid

colorModel = CI.ColorizeImageTorch(Xd=256)

colorModel.prep_net(path='./models/pytorch/caffemodel.pth')

app = Flask(__name__)

def put_point(input_ab, mask, loc, p, val):
    input_ab[:, loc[0] - p: loc[0] + p + 1, loc[1] - p : loc[1] + p + 1] = np.array(val)[:, np.newaxis, np.newaxis]
    mask[:, loc[0] - p : loc[0] + p + 1, loc[1] - p : loc[1] + p + 1] = 1
    return (input_ab, mask)

@app.route('/ideepcolor', methods=['POST'])
def ideepcolor():
    try:
        image = request.files['image']
        color = request.form['color']
        positionX = request.form['positionX']
        positionY = request.form['positionY']
    except:
        return Response("Bad Request", status=400)
    
    filename = uuid.uuid1()
    image.save(f'./input/{filename}.{image.content_type.split("/")[-1]}')

    colorModel.load_image(f'./input/{filename}.{image.content_type.split("/")[-1]}')

    input_ab = np.zeros((2, 256, 256))
    mask = np.zeros((1, 256, 256))

    rgbArray = ImageColor.getrgb(color)
    rgbNumpyArr = np.array((rgbArray[0], rgbArray[1], rgbArray[2])).astype('uint8')
    labArr = skiColor.rgb2lab(rgbNumpyArr[np.newaxis, np.newaxis, :]).flatten()

    (input_ab, mask) = put_point(input_ab, mask, [int(positionY), int(positionX)], 3, [labArr[1], labArr[2]])
    colorModel.net_forward(input_ab, mask)
    img_out_fullres = colorModel.get_img_fullres()
    pilImage = Image.fromarray(np.uint8(img_out_fullres)).convert('RGB')
    result = BytesIO()
    pilImage.save(result, format=image.content_type.split("/")[-1])
    result.seek(0)
    os.remove(f'./input/{filename}.{image.content_type.split("/")[-1]}')

    return send_file(result, mimetype=f'image/{image.content_type.split("/")[-1]}')

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/healthz', methods=['GET'])
def healthz():
    return 'ok'

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="5000")