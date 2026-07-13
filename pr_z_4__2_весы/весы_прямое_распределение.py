#Установка пользовательских весов в полносвязном слое и прямое распространение
import numpy as np
from keras import models,layers 

model =  models.Sequential([
    layers.Dense(3,input_shape=(5,),activation='relu')
])

# Все весоь 0.5
weights = np.full((5,3),0.5)
biases = np.zeros(3)
models.layers[0].set_weights([weights,biases])

x= np.array ([1,2,3,4,5],[0,1,0,1,0])
y= models.predict(x)

print("enters:",x)
print("outputs:",y)

