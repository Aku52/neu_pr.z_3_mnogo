#Классификация рукописных цифр MNIST с помощью полносвязной нейронной сети

from keras.datasets import mnist
import tensorflow
from keras import models, layers
from keras.utils import to_categorical
import matplotlib as plt


# ЗАГРУЗКА ДАННЫХ MNIST
(x_train, y_train), (x_test,y_test) = mnist.load_data() # разделение данныз на тренировку и тест

x_train = x_train.reshape((60000, 28*28)).astype("float32")/255 # 255 - обычная нормализация для пикселей (6000 из-й)
x_test = x_test.reshape((10000, 28*28)).astype("float32")/255


# ONE-HOT ENCODING МЕТОК
y_train = to_categorical(y_train)                                     # [5] → [0,0,0,0,0,1,0,0,0,0]
y_test = to_categorical(y_test)
print(y_train[0])                                                     # вывод первой метки


model = models.Sequential([# последовательные слои модели (исп-е Sequential)
    layers.Dense(512,activation='relu', input_shape = (784,)), # скрытый слой(Dense- полнозначный слой(внашем случае 512),
    #input_shape-размер входного батча), relu - обеспечение нелинейности (для упрощения обучения)

    layers.Dense(10,activation='softmax')# выходной слой(10 нейронов(в нашем mnist всего лишь 10 возможных классов,
     # поэтому на каждый класс по однойц связи (от 0 до 9), softmax = вероятность))
])

# КОМПИНАЦИЯ
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) #loss - функция потерь(есть много других вариантов, но эта просто самая легковесная)
# adam - умная версия градиентного спуска

history = model.fit(x_train, y_train, epochs=5, batch_size = 128, valitation_split = 0.1)# batch_size = батчи, котрые поступают на вход(делим на 128)
# history.history['loss','val_loss','accuracy','val_accuracy']

# ПОСТРОЕНИЕ ГРАФИКОВ ПОТЕРЬ
plt.plot(history.history['loss'], label="train_loss")                 # график потерь на обучении
plt.plot(history.history['val_loss'], label="val_loss")               # график потерь на валидации
plt.legend()                                                          # легенда
plt.show()                                                            # отображение графика
