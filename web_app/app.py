from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("./models")

"""Create and configure an instance of the Flask application"""
app = Flask(__name__)
data = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

@app.route('/', methods=['GET', 'POST'])


def predict():
    # data = request.json(force=True)
    result = np.exp(model.predict(data))
    return jsonify(result)

if __name__ == '__main__':
    app.run()
