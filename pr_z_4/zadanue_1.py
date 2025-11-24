import numpy as np
from keras import models,layers 

model =  models.Sequential([
    layers.Dense(10,input_shape=(5,))
])

# Получение весов и смещение (сдвигает порорг решаемых решений)
# Веса - какую роль играет 
weights,biases = models.layers[0].get_weights()

print (weights.shape)
print (biases.shape)
print (weights[:3,:5]) # Веса 
print (biases[:10]) # Первые 10 сдвигов

