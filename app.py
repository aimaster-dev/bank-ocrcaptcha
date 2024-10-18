from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
from flask import jsonify

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import model_from_json
import base64
import time

# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
class mbbank():
    img_width = 320
    img_height = 80
    max_length = 6
    characters_mbbank = ['2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'G', 'H', 'K', 'M', 'N', 'P', 'Q', 'U', 'V', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'k', 'm', 'n', 'p', 'q', 't', 'u', 'v', 'y', 'z']
    # Mapping characters to integers
    char_to_num = layers.StringLookup(
        vocabulary=list(characters_mbbank), mask_token=None
    )
class bidv():
    img_width = 145
    img_height = 50
    max_length = 6
    characters_mbbank = ['2', '3', '4', '5', '6', '7', '8', '9', 'V', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z']
    # Mapping characters to integers
    char_to_num = layers.StringLookup(
        vocabulary=list(characters_mbbank), mask_token=None
    )

    # Mapping integers back to original characters
    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )
class eximbank():
    img_width = 145
    img_height = 50
    max_length = 6
    characters_mbbank = ['2', '3', '4', '5', '6', '7', '8', '9', 'V', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z']
    # Mapping characters to integers
    char_to_num = layers.StringLookup(
        vocabulary=list(characters_mbbank), mask_token=None
    )

    # Mapping integers back to original characters
    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )
class vietcombank(): 
    img_width = 155
    img_height = 50
    max_length = 6
    characters_mbbank = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    # Mapping characters to integers
    char_to_num = layers.StringLookup(
        vocabulary=list(characters_mbbank), mask_token=None
    )

    # Mapping integers back to original characters
    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )
def LoadModel(file):
    # JSON format
    xfile = os.path.splitext(file)
    if (xfile[1] == ".json"):
        with open(file, "r") as json:
            json_model = json.read()
        model = keras.models.model_from_json(json_model)
        model.load_weights(xfile[0] + ".wgt")
    # ONNX format
    elif (xfile[1] == ".onnx"):
        raise Exception("LoadModel; ONNX format not supported yet")
        model = None
    # TF/Keras format
    else:
        model = keras.models.load_model(file, custom_objects={'leaky_relu': tf.nn.leaky_relu})
    return model

model_mbbank = LoadModel("mbbank/mbbank.json")
model_bidv = LoadModel("bidv/bidv.json")
model_eximbank = LoadModel("eximbank/eximbank.json")
model_vietcombank = LoadModel("vietcombank/vietcombank.json")
# A utility function to decode the output of the network

def decode_batch_predictions(pred,bank="mbbank"):
    if bank == "mbbank":
        bank_class = mbbank
    if bank == "bidv":
        bank_class = bidv
    if bank == "vietcombank":
        bank_class = vietcombank
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :bank_class.max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(bank_class.num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text
  
def encode_base64x(base64,bank="mbbank"):
    if bank == "mbbank":
        bank_class = mbbank
    if bank == "bidv":
        bank_class = bidv
    if bank == "eximbank":
        bank_class = eximbank
    if bank == "vietcombank":
        bank_class = vietcombank
    img = tf.io.decode_base64(base64)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [bank_class.img_height, bank_class.img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    return {"image": img}

@app.route("/api/captcha/mbbank", methods=["POST"])
@cross_origin(origin='*')
def mbbank_captcha_solver():
    content = request.json
    start_time = time.time()
    imgstring = content['base64']
    image_encode = encode_base64x(imgstring.replace("+", "-").replace("/", "_"),"mbbank")["image"]
    listImage = np.array([image_encode])
    preds = model_mbbank.predict(listImage)
    pred_texts = decode_batch_predictions(preds,"mbbank")
    captcha = pred_texts[0].replace('[UNK]', '').replace('-', '')
    response = jsonify(status = "success",captcha = captcha)

    return response

@app.route("/api/captcha/bidv", methods=["POST"])
@cross_origin(origin='*')
def bidv_captcha_solver():
    content = request.json
    start_time = time.time()
    imgstring = content['base64']
    image_encode = encode_base64x(imgstring.replace("+", "-").replace("/", "_"),"bidv")["image"]
    listImage = np.array([image_encode])
    preds = model_bidv.predict(listImage)
    pred_texts = decode_batch_predictions(preds,"bidv")
    captcha = pred_texts[0].replace('[UNK]', '').replace('-', '')
    response = jsonify(status = "success",captcha = captcha)

    return response

@app.route("/api/captcha/eximbank", methods=["POST"])
@cross_origin(origin='*')
def eximbank_captcha_solver():
    content = request.json
    start_time = time.time()
    imgstring = content['base64']
    image_encode = encode_base64x(imgstring.replace("+", "-").replace("/", "_"),"eximbank")["image"]
    listImage = np.array([image_encode])
    preds = model_eximbank.predict(listImage)
    pred_texts = decode_batch_predictions(preds,"eximbank")
    captcha = pred_texts[0].replace('[UNK]', '').replace('-', '')
    response = jsonify(status = "success",captcha = captcha)

    return response

@app.route("/api/captcha/vietcombank", methods=["POST"])
@cross_origin(origin='*')
def vietcombank_captcha_solver():
    content = request.json
    start_time = time.time()
    imgstring = content['base64']
    image_encode = encode_base64x(imgstring.replace("+", "-").replace("/", "_"),"vietcombank")["image"]
    listImage = np.array([image_encode])
    preds = model_vietcombank.predict(listImage)
    pred_texts = decode_batch_predictions(preds,"vietcombank")
    captcha = pred_texts[0].replace('[UNK]', '').replace('-', '')
    response = jsonify(status = "success",captcha = captcha)

    return response

# Chạy server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8277')