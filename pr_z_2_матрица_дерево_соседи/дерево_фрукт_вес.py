# Дерево решений для классификации фруктов по весу

# ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# СОЗДАНИЕ ДАННЫХ
data = pd.DataFrame({
    "fruit": ["apple", "orange", "banan", "pineapple", "orange", "apple"],  # категориальный признак
    "weight": [100, 80, 50, 90, 125, 150],                                  # числовой признак
    "label": ["low", "low", "average", "low", "hight", "hight"]             # целевая переменная
})

# ONE-HOT ENCODING ДЛЯ КАТЕГОРИАЛЬНОГО ПРИЗНАКА
encoder = OneHotEncoder()                                 # создаём кодировщик
x_encoded = encoder.fit_transform(data[["fruit"]])        # преобразуем фрукты в бинарные столбцы

# ОБЪЕДИНЕНИЕ ПРИЗНАКОВ
x = np.hstack([x_encoded.toarray(), data[["weight"]].values])  # кодированные фрукты + вес

# ЦЕЛЕВАЯ ПЕРЕМЕННАЯ
y = data["label"]                                             # метки (low, average, hight)

# ОБУЧЕНИЕ ДЕРЕВА РЕШЕНИЙ
clf = DecisionTreeClassifier()                                # создаём классификатор
clf.fit(x, y)                                                 # обучаем на признаках и метках

# ПРОГНОЗ ДЛЯ НОВОГО ОБЪЕКТА
new = encoder.transform([["apple"]]).toarray()                # кодируем "apple"
new_data = np.hstack([new, [[115]]])                          # добавляем вес 115

print(clf.predict(new_data))                                  # предсказание (low/average/hight)