# Построить модель сохранить веса обучить одну
#  эпоху снова сохранить. Сравнить, какие веса изменились сильнее.

# ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
import numpy as np
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical

# ЗАГРУЗКА ДАННЫХ MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# БЕРЁМ ПЕРВЫЕ 200 ИЗОБРАЖЕНИЙ, НОРМАЛИЗУЕМ И ПРЕОБРАЗУЕМ В ВЕКТОР
x_train = x_train[:200].reshape((200, 28*28)).astype("float32") / 255   # 28×28 = 784
y_train = y_train[:200]                                                # первые 200 меток

# СОЗДАНИЕ МОДЕЛИ
model = models.Sequential([
    layers.Dense(3, activation='relu', input_shape=(28*28,)),           # скрытый слой (3 нейрона, ReLU)
    layers.Dense(32, activation='relu'),                                # скрытый слой (32 нейрона, ReLU)
    layers.Dense(10, activation='softmax')                              # выходной слой (10 классов, Softmax)
])

# КОМПИЛЯЦИЯ
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'                              # для целочисленных меток
)

# СОХРАНЕНИЕ ВЕСОВ ДО ОБУЧЕНИЯ
w_before, b_before = model.layers[0].get_weights()                     # веса первого слоя до обучения

# ОБУЧЕНИЕ (1 ЭПОХА)
model.fit(
    x_train, y_train,
    epochs=1,                                                          # 1 эпоха
    batch_size=32,                                                     # размер батча
    verbose=1
)

# СОХРАНЕНИЕ ВЕСОВ ПОСЛЕ ОБУЧЕНИЯ
w_after, b_after = model.layers[0].get_weights()                       # веса первого слоя после обучения

# ВЫЧИСЛЕНИЕ ИЗМЕНЕНИЙ
delta_w = w_after - w_before                                           # разница весов

# ВЫВОД РЕЗУЛЬТАТОВ
print("Среднее изменение весов:", np.mean(np.abs(delta_w)))            # среднее абсолютное изменение
print("Максимальное изменение весов:", np.max(np.abs(delta_w)))        # максимальное абсолютное изменение