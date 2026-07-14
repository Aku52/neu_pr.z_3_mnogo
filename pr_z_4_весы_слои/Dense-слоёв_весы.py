# Построить модель из 3 Dense-слоёв и вывести формы весов каждого

# ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
import numpy as np
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical

# ЗАГРУЗКА ДАННЫХ MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# БЕРЁМ ПЕРВЫЕ 200 ИЗОБРАЖЕНИЙ, НОРМАЛИЗУЕМ И ПРЕОБРАЗУЕМ В ВЕКТОР
x_train = x_train[:200].reshape((200, 28*28)).astype("float32") / 255   # 28×28 = 784 признака

# БЕРЁМ ПЕРВЫЕ 200 МЕТОК
y_train = y_train[:200]

# СОЗДАНИЕ МОДЕЛИ
model = models.Sequential([
    layers.Dense(3, activation='relu', input_shape=(28*28,)),           # скрытый слой (3 нейрона)
    layers.Dense(32, activation='relu'),                                # скрытый слой (32 нейрона)
    layers.Dense(10, activation='softmax')                              # выходной слой (10 классов)
])

# КОМПИЛЯЦИЯ
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'                              # для целочисленных меток
)

# ПОЛУЧЕНИЕ ВЕСОВ ДО ОБУЧЕНИЯ
w_before, b_before = model.layers[0].get_weights()                     # веса и смещения первого слоя

# ОБУЧЕНИЕ МОДЕЛИ
model.fit(
    x_train, y_train,
    epochs=3,                                                          # 3 эпохи
    batch_size=32,                                                     # размер батча
    verbose=1
)

# ПОЛУЧЕНИЕ ВЕСОВ ПОСЛЕ ОБУЧЕНИЯ
w_after, b_after = model.layers[0].get_weights()                       # обновлённые веса первого слоя

# ВЫВОД ФОРМ ВЕСОВ ДЛЯ КАЖДОГО СЛОЯ
for i, layer in enumerate(model.layers):
    w, b = layer.get_weights()
    print(f"Слой {i}: форма весов {w.shape}")                          # (784,3), (3,32), (32,10)