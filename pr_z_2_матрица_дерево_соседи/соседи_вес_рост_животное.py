#Классификация животных по весу и росту с помощью метода k-ближайших соседей (KNN)

# ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# СОЗДАНИЕ ДАННЫХ
data = pd.DataFrame({
    "animal": ["cat", "cat", "dog", "dog", "cat", "dog"],  # вид животного
    "weight": [1, 1.5, 50, 90, 5, 30],  # вес
    "height": [10, 15.50, 45, 5, 80],   # рост
    "label": ["home", "home", "service", "service", "home", "service"]  # целевая переменная
})

# ONE-HOT ENCODING ДЛЯ КАТЕГОРИАЛЬНОГО ПРИЗНАКА
encoder = OneHotEncoder()    # создаём кодировщик
x_encoded = encoder.fit_transform(data[["animal"]]) # преобразуем animal в бинарные столбцы

# ОБЪЕДИНЕНИЕ ПРИЗНАКОВ
x = np.hstack([x_encoded.toarray(), data[["weight", "height"]].values])  # объединение one-hot признаков с числовыми

# ЦЕЛЕВАЯ ПЕРЕМЕННАЯ
y = data["label"]

# ОБУЧЕНИЕ KNN
knn = KNeighborsClassifier(n_neighbors=3) # создаём KNN с 3 соседями
knn.fit(x, y) # обучаем модель

# ПРОГНОЗ ДЛЯ НОВОГО ОБЪЕКТА
new = encoder.transform([["dog"]]).toarray() # кодируем "dog"
new_data = np.hstack([new, [[47, 28]]]) # добавляем вес 47 и рост 28

print(knn.predict(new_data)) # предсказание (home/service)