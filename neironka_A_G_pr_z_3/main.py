from keras.datasets import cifar10
from keras import models, layers
from keras.utils import to_categorical


(x_train, y_train), (x_test,y_test) = mnist.load_data() # разделение данныз на тренировку и тест
x_train,x_test = x_train/255.0, x_test/255.0
# Практически тоже самое
#x_train = x_train.reshape((60000, 28*28)).astype("float32")/255 # 255 - обычная нормализация для пикселей (6000 из-й)
#x_test = x_test.reshape((10000, 28*28)).astype("float32")/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



model = models.Sequential([# последовательные слои модели (исп-е Sequential)
    layers.Conv2D(32,(3,3),activation='relu', input_shape = (32,32,3)),
    #input_shape-размер входного батча), relu - обеспечение нелинейности (для упрощения обучения)
    
    layers.MaxPooling2D((2,2))
    layers.Conv2D(64,(3,3),activation='relu')
    layers.MaxPooling2D((2,2))
    layers.Flatten(), # Flatten распремляет
    layers.Dense(64,activation='relu')
    layers.Dense(10,activation='softmax')
])

model.compile(
    optimizer='adam', # adam - умная версия градиентного спуска
    loss = 'categorial_crossentropy', #loss - функция потерь(есть много других вариантов, но эта просто самая легковесная)
    metrics = ['accurasy']
)


history= model.fit(
    x_train, 
    y_train, 
    epochs=10, 
    batch_size = 64, # batch_size = батчи, котрые поступают на вход(делим на 128)
    valitation_split = 0.1)
