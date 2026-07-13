# Построить модель сохранить веса обучить одну
#  эпоху снова сохранить. Сравнить, какие веса изменились сильнее.


# Импорт библиотек
import numpy as np
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:200].reshape((200, 28*28)).astype("float32") / 255
y_train = y_train[:200]

# Создание модели
model = models.Sequential([
    layers.Dense(3, activation='relu', input_shape=(28*28,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)

# Первое сохранение весо
w_before, b_before = model.layers[0].get_weights()
# Тренировка весов
model.fit(
        x_train, y_train, 
        epochs=1, 
        batch_size=32, 
        verbose=1)
# Второе сохранение весов
w_after, b_after = model.layers[0].get_weights()
delta_w = w_after -w_before
# Сравнение и сохранение
print("Среднее изменение весов:", np.mean(np.abs(delta_w)))
print("Максимальное изменение весов:", np.max(np.abs(delta_w)))


