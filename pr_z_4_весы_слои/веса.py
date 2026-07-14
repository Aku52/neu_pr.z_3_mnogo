#"Изучение работы полносвязного слоя нейронной сети:
#  установка весов и получение выходных значений"

import numpy as np
from keras import models,layers 

model =  models.Sequential([
    layers.Dense(3,input_shape=(5,),activation='relu')        # полносвязный слой: 5 входов, 3 нейрона, ReLU
])

# УСТАНОВКА ПОЛЬЗОВАТЕЛЬСКИХ ВЕСОВ
# Все весоь 0.5
weights = np.full((5, 3), 0.5) # матрица 5×3, все значения 0.5
biases = np.zeros(3)     # смещения = 0
model.layers[0].set_weights([weights, biases])  

# ВХОДНЫЕ ДАННЫЕ (2 примера по 5 признаков)
x = np.array([[1, 2, 3, 4, 5], [0, 1, 0, 1, 0]])                     
# ПРЯМОЕ РАСПРОСТРАНЕНИЕ
y = model.predict(x)        
# ПОЛУЧЕНИЕ ВЕСОВ ПОСЛЕ ОБУЧЕНИЯ
weights,biases = models.layers[0].get_weights()

# ВЫВОД РАЗМЕРНОСТЕЙ
print(weights.shape)                                                 # (5, 3)
print(biases.shape)                                                  # (3,)

# ВЫВОД ЧАСТИ ВЕСОВ
print (weights[:3,:5]) # Веса # первые 3 строки, все 5 столбцов
# ВЫВОД СМЕЩЕНИЙ
print (biases[:10]) # Первые 10 сдвигов # все смещения (3 значения)
    