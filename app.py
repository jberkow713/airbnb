import bz2
from flask import Flask, request, jsonify
import _pickle as cPickle
import numpy as np


def decompress_pickle(file):
    model = bz2.BZ2File(file, 'rb')
    model = cPickle.load(model)
    return model

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    model = decompress_pickle('rf_model.pbz2')
    result = np.exp(model.predict(data)).tolist()
    return jsonify(result)


if __name__ == '__main__':
    app.run()


# from flask import Flask, jsonify, request
# import tensorflow as tf
# import numpy as np


# model = tf.keras.models.load_model("models/saved_model.pb")

# """Create and configure an instance of the Flask application"""
# app = Flask(__name__)
# data = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

# @app.route('/', methods=['GET', 'POST'])


# def predict():
#     # data = request.json(force=True)
#     result = np.exp(model.predict(data))
#     return jsonify(result)