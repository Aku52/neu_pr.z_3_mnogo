import numpy as np
from keras import models,layers 

model =  models.Sequential([
    layers.Dense(3,input_shape=(5,),activation='relu')
])

# Все весоь 0.5
weights = np.full((5,3),0.5)
biases = np.zeros(3)
models.layers[0].set_weights([W,b])
x= np.array ([1,2,3,4,5],[0,1,0,1,0])
y= models.predict(x)



weights,biases = models.layers[0].get_weights()

print (weights.shape)
print (biases.shape)
print (weights[:3,:5]) # Веса 
print (biases[:10]) # Первые 10 сдвигов