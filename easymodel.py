from os import path
import json
import tensorflow as tf
from tensorflow.keras import layers, optimizers, constraints, regularizers
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime

print("Tensorflow", tf.__version__)

selected_columns = ['x_right_hand_0', 'y_right_hand_0', 'z_right_hand_0',
                    'x_right_hand_1', 'y_right_hand_1', 'z_right_hand_1',
                    'x_right_hand_2', 'y_right_hand_2', 'z_right_hand_2',
                    'x_right_hand_3', 'y_right_hand_3', 'z_right_hand_3',
                    'x_right_hand_4', 'y_right_hand_4', 'z_right_hand_4',
                    'x_right_hand_5', 'y_right_hand_5', 'z_right_hand_5',
                    'x_right_hand_6', 'y_right_hand_6', 'z_right_hand_6',
                    'x_right_hand_7', 'y_right_hand_7', 'z_right_hand_7',
                    'x_right_hand_8', 'y_right_hand_8', 'z_right_hand_8',
                    'x_right_hand_9', 'y_right_hand_9', 'z_right_hand_9',
                    'x_right_hand_10', 'y_right_hand_10', 'z_right_hand_10',
                    'x_right_hand_11', 'y_right_hand_11', 'z_right_hand_11',
                    'x_right_hand_12', 'y_right_hand_12', 'z_right_hand_12',
                    'x_right_hand_13', 'y_right_hand_13', 'z_right_hand_13',
                    'x_right_hand_14', 'y_right_hand_14', 'z_right_hand_14',
                    'x_right_hand_15', 'y_right_hand_15', 'z_right_hand_15',
                    'x_right_hand_16', 'y_right_hand_16', 'z_right_hand_16',
                    'x_right_hand_17', 'y_right_hand_17', 'z_right_hand_17',
                    'x_right_hand_18', 'y_right_hand_18', 'z_right_hand_18',
                    'x_right_hand_19', 'y_right_hand_19', 'z_right_hand_19',
                    'x_right_hand_20', 'y_right_hand_20', 'z_right_hand_20',
                    ]

selected_columns_dict = {"selected_columns": selected_columns}
inference_args_path = path.join(working_dir, 'inference_args.json')
with open(inference_args_path, "w") as f:
    json.dump(selected_columns_dict, f)

import tensorflow as tf

input_dim = len(selected_columns)
output_dim = 59

inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32, name='inputs')
x = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
outputs = tf.keras.layers.Dense(output_dim, activation=tf.nn.relu, name='outputs')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss="sparse_categorical_crossentropy",
                            metrics="accuracy")
#model.summary()

tf_model_path = path.join(working_dir, 'tf_model')
model.save(tf_model_path)

# Convert the model
tflite_model_path = path.join(working_dir, 'model.tflite')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
