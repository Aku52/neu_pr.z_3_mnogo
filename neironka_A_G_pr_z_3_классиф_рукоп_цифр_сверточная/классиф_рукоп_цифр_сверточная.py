# Классификация рукописных цифр MNIST с помощью свёрточной нейронной сети

from keras.datasets import cifar10
from keras import models, layers
from keras.utils import to_categorical


# ЗАГРУЗКА ДАННЫХ MNIST (рукописные цифры)
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # разделение на тренировку и тест
x_train, x_test = x_train / 255.0, x_test / 255.0  # нормализация пикселей (0-255 → 0-1)
# Практически тоже самое
#x_train = x_train.reshape((60000, 28*28)).astype("float32")/255 # 255 - обычная нормализация для пикселей (6000 из-й)
#x_test = x_test.reshape((10000, 28*28)).astype("float32")/255

# ПРЕОБРАЗОВАНИЕ МЕТОК В ONE-HOT ENCODING
# [5] → [0,0,0,0,0,1,0,0,0,0]
y_train = to_categorical(y_train)  # 10 классов → 10 бинарных столбцов
y_test = to_categorical(y_test)

# СОЗДАНИЕ МОДЕЛИ (СВЁРТОЧНАЯ НЕЙРОСЕТЬ)
model = models.Sequential([ # последовательные слои модели (исп-е Sequential)
    layers.Conv2D(32,(3,3),activation='relu', input_shape = (32,32,3)),
    #input_shape-размер входного батча), relu - обеспечение нелинейности (для упрощения обучения)
    
    # ПОДВЫБОРКА (уменьшение размерности)
    layers.MaxPooling2D((2, 2)),  # берёт максимум из окна 2×2 (уменьшает размер вдвое)

    # СВЁРТОЧНЫЙ СЛОЙ №2
    layers.Conv2D(64, (3, 3), activation='relu'),
    # 64 фильтра размером 3×3

    # ПОДВЫБОРКА №2
    layers.MaxPooling2D((2, 2)),

    # ПРЕОБРАЗОВАНИЕ В ОДНОМЕРНЫЙ ВЕКТОР
    layers.Flatten(),  # "распрямляет" многомерный массив в одномерный

    # ПОЛНОСВЯЗНЫЙ СЛОЙ
    layers.Dense(64, activation='relu'),  # 64 нейрона с ReLU

    # ВЫХОДНОЙ СЛОЙ
    layers.Dense(10, activation='softmax')  # 10 классов (цифры 0-9)
    # Softmax: вероятности принадлежности к каждому классу (сумма = 1)
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


# КОМПИЛЯЦИЯ МОДЕЛИ
model.compile(
    optimizer='adam',  # оптимизатор (адаптивный градиентный спуск)
    loss='categoriсal_crossentropy',  # функция потерь для многоклассовой классификации
    metrics=['accuraсy']  # метрика качества (точность)
)

# ОБУЧЕНИЕ МОДЕЛИ
history = model.fit(
    x_train,  # тренировочные данные (изображения)
    y_train,  # тренировочные метки (цифры)
    epochs=10,  # количество эпох (полных проходов по данным)
    batch_size=64,  # размер батча (сколько примеров за 1 шаг)
    validation_split=0.1  # доля данных для валидации (10%)
)