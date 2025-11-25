# Построить модель из 3 Dense-слоёв и вывести формы весов каждого

import numpy as np
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:200].reshape((200, 28*28)).astype("float32") / 255
y_train = y_train[:200]


model = models.Sequential([
    layers.Dense(3, activation='relu', input_shape=(28*28,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)

w_before, b_before = model.layers[0].get_weights()

model.fit(
        x_train, y_train, 
        epochs=3, 
        batch_size=32, 
        verbose=1)

w_after, b_after = model.layers[0].get_weights()

for i,layer in enumerate (model.layers):
    w, b = layer.get_weights()
    print(f"Слой {i}: форма весов {w.shape}")
