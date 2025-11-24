import numpy as np
from keras import models,layers 
from keras.datasets import mnist 
from keras.utils import to_categorical

(x_train, y_train), (x_test,y_test) = mnist.load_data() 
x_train = x_train[:200].reshape((200, 28*28)).astype("float32")/255 
x_test =to_categorical(y_train[:200])
#x_train[:200]- первые 200 эл, reshape((200, 28*28))-размер,astype("float32") - приводение к типу

model= models.Sequential([
        layers.Dense(64, activation='relu',input_shape=(28*28,)), 
        layers.Dense(10,activation='softmax') 
    ])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
    )

w_before,b_before= models.layers[0].set_weights()

# На каждой эпохе веса меняются 
model.fit(
        x_train, y_train, 
        epochs=1, 
        batch_size=32, 
        verbose=1)

w_after,b_after = models.layers[0].set_weights()
delta_w = w_after -w_before

print("Максимальное значение веса:",np.max(np.abs(delta_w)))
print("Среднее значение веса:",np.mean(np.abs(delta_w))) 
# np.mean(np.abs(delta_w)), abs- модуль,mean-среднее арифмитическое, delta_w- изменения