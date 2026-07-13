#Исследование различных инициализаторов весов в полносвязной нейронной сети
import numpy as np
from keras import models,layers 

#Чтото про веса
intializators= ['random_normal','he_normal','glorot_uniform']
for init in intializators:
    model= models.Sequential([
    layers.Dense(64, kerner_intializator=init,input_shape=(100,)), 
    layers.Dense(10,activation='softmax') 
    ])
    w,b= models.layers[0].set_weights()
    print(f'intializer:{init}')
    print(f'mean:{np.mean(w)},std: {np.std(w)}')
    print(f'min:{np.min(w)},max: {np.max(w)}')
