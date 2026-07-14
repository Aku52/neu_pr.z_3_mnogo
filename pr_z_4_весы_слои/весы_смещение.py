#Изучение структуры полносвязного слоя: получение весов и смещений"

# ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
import numpy as np
from keras import models, layers

# СОЗДАНИЕ МОДЕЛИ
model = models.Sequential([
    layers.Dense(10, input_shape=(5,))   # полносвязный слой: 5 входов, 10 нейронов (без активации)
])

# ПОЛУЧЕНИЕ ВЕСОВ И СМЕЩЕНИЙ
weights, biases = model.layers[0].get_weights()             
# ВЫВОД РАЗМЕРНОСТЕЙ
print(weights.shape)         # (5, 10) — 5 входов, 10 нейронов
print(biases.shape)        # (10,) — смещение для каждого нейрона

# ВЫВОД ЧАСТИ ВЕСОВ
print(weights[:3, :5])    # первые 3 строки, первые 5 столбцов

# ВЫВОД СМЕЩЕНИЙ
print(biases[:10])     # все 10 смещений
