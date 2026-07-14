#"Классификация типа устройства (стандартный/премиум) на основе категории устройства,
#  бренда и цены с помощью дерева решений и KNN"

# ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# СОЗДАНИЕ ДАННЫХ
data = pd.DataFrame({
    "devices": ["phone", "comp", "comp", "notebook", "phone", "notebook", "tablet", "tablet"],  # тип устройства
    "brand": ["samsung", "LG", "apple", "honor", "apple", "huawei", "samsumg", "redmi"],        # бренд (опечатка в samsumg)
    "price": [1200, 4500, 12000, 4000, 3600, 2350, 4000, 2000],                               # цена
    "type": ["standart", "standart", "premium", "premium", "premium", "standart", "premium", "standart"]  # целевая переменная
})

# ONE-HOT ENCODING ДЛЯ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
encoder = OneHotEncoder()                                             # создаём кодировщик
x_encoded = encoder.fit_transform(data[["devices", "brand"]])         # преобразуем в бинарные столбцы

# ОБЪЕДИНЕНИЕ ПРИЗНАКОВ
X = np.hstack([x_encoded.toarray(), data[["price"]].values])          # кодированные признаки + цена

# ЦЕЛЕВАЯ ПЕРЕМЕННАЯ
y = data["type"]                                                      # метки (standart, premium)

# ПОДГОТОВКА НОВОГО ОБЪЕКТА ДЛЯ ПРОГНОЗА
new = encoder.transform([["comp", "LG"]]).toarray()                   # кодируем comp и LG

# МОДЕЛЬ 1: ДЕРЕВО РЕШЕНИЙ
tree = DecisionTreeClassifier(max_depth=2).fit(X, y)                  # обучаем дерево (глубина 2)
new_data = np.hstack([new, [[5000]]])                                 # добавляем цену 5000
print(f"Decisison tree: {tree.predict(new_data)}\n")                  # прогноз дерева


# МОДЕЛЬ 2: KNN (K-БЛИЖАЙШИХ СОСЕДЕЙ)
knn = KNeighborsClassifier(n_neighbors=3)                             # создаём KNN с 3 соседями
knn.fit(X, y)                                                         # обучаем KNN

print(knn.predict(new_data))                                          # прогноз KNN