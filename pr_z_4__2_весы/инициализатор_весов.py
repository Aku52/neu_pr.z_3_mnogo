#Исследование различных инициализаторов весов в полносвязной нейронной сети

# ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
import numpy as np
from keras import models, layers

# СПИСОК ИНИЦИАЛИЗАТОРОВ ДЛЯ ТЕСТИРОВАНИЯ
intializators = ['random_normal', 'he_normal', 'glorot_uniform']

# ЦИКЛ ПО ВСЕМ ИНИЦИАЛИЗАТОРАМ
for init in intializators:
    # СОЗДАНИЕ МОДЕЛИ
    model = models.Sequential([
        layers.Dense(64, kernel_initializer=init, input_shape=(100,)),  
        layers.Dense(10, activation='softmax')
    ])
    
    # ПОЛУЧЕНИЕ ВЕСОВ ПЕРВОГО СЛОЯ
    w, b = model.layers[0].get_weights()                               
    
    # ВЫВОД СТАТИСТИКИ ВЕСОВ
    print(f'intializer: {init}')
    print(f'mean: {np.mean(w)}, std: {np.std(w)}')
    print(f'min: {np.min(w)}, max: {np.max(w)}')