#Установка пользовательских весов в полносвязном слое и прямое распространение

# ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
import numpy as np
from keras import models, layers

# СОЗДАНИЕ МОДЕЛИ
model = models.Sequential([
    layers.Dense(3, input_shape=(5,), activation='relu')   # полносвязный слой: 5 входов, 3 нейрона, ReLU
])

# УСТАНОВКА ПОЛЬЗОВАТЕЛЬСКИХ ВЕСОВ
weights = np.full((5, 3), 0.5)       # матрица 5×3, все значения 0.5
biases = np.zeros(3)             # смещения = 0
model.layers[0].set_weights([weights, biases])               

# ВХОДНЫЕ ДАННЫЕ (2 примера по 5 признаков)
x = np.array([[1, 2, 3, 4, 5], [0, 1, 0, 1, 0]])                     

# ПРЯМОЕ РАСПРОСТРАНЕНИЕ (FORWARD PASS)
y = model.predict(x)                                              

# ВЫВОД РЕЗУЛЬТАТОВ
print("enters:", x)
print("outputs:", y)           