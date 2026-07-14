#Изучение изменения весов нейронной сети в процессе обучения на ограниченной выборке MNIST

# ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
import numpy as np
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical

# ЗАГРУЗКА ДАННЫХ MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# БЕРЁМ ПЕРВЫЕ 200 ИЗОБРАЖЕНИЙ ДЛЯ УСКОРЕНИЯ
x_train = x_train[:200].reshape((200, 28*28)).astype("float32") / 255   # нормализация и reshape
x_test = to_categorical(y_train[:200])            

# СОЗДАНИЕ МОДЕЛИ
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(28*28,)),         # скрытый слой (64 нейрона, ReLU)
    layers.Dense(10, activation='softmax')                             # выходной слой (10 классов, Softmax)
])

# КОМПИЛЯЦИЯ
model.compile(
    optimizer='adam', # adam - умная версия градиентного спуска
    loss='sparse_categorical_crossentropy'    # для целочисленных меток (без one-hot)
)

# ПОЛУЧЕНИЕ ВЕСОВ ДО ОБУЧЕНИЯ
w_before, b_before = model.layers[0].get_weights()       
model.fit(
    x_train, y_train,
    epochs=1,
    batch_size=32,
    verbose=1
)

# ПОЛУЧЕНИЕ ВЕСОВ ПОСЛЕ ОБУЧЕНИЯ
w_after, b_after = model.layers[0].get_weights()                  

# ВЫЧИСЛЕНИЕ ИЗМЕНЕНИЙ ВЕСОВ
delta_w = w_after - w_before                                          # разница весов до и после

# ВЫВОД РЕЗУЛЬТАТОВ
print("Максимальное значение веса:", np.max(np.abs(delta_w)))         # максимальное изменение
print("Среднее значение веса:", np.mean(np.abs(delta_w)))             # среднее изменение