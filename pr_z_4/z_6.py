# Построить модель из 3 Dense-слоёв и вывести формы весов каждого
# Усложнение:Сравнить, как изменится форма при другом размере входного вектора
import numpy as np
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# уменьшаем каждое изображение до 24x24
x_train = tf.image.resize(x_train[..., tf.newaxis], (24, 24)).numpy()

x_train= x_train[:200].reshape((200, 24*24)).astype("float32") / 255

y_train = y_train[:200]


model = models.Sequential([
    layers.Dense(3, activation='relu', input_shape=(24*24,)),
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

print (w_before.shape)
print (b_before.shape)