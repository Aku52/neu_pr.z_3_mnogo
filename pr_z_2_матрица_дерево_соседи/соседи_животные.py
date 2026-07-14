#"Сравнение классификации животных с помощью дерева решений и"
#" метода k-ближайших соседей (KNN)"

# ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# СОЗДАНИЕ ДАННЫХ
data = pd.DataFrame({
    "animal": ["cat", "cat", "dog", "dog", "cat", "dog"],          # вид животного
    "weight": [1, 1.5, 50, 90, 5, 30],                             # вес
    "height": [10, 15.50, 45, 5, 80],                              # рост
    "label": ["home", "home", "service", "service", "home", "service"]  # целевая переменная
})

# ONE-HOT ENCODING
encoder = OneHotEncoder()                                          # создаём кодировщик
x_encoded = encoder.fit_transform(data[["animal"]])                # преобразуем animal в бинарные столбцы

# ОБЪЕДИНЕНИЕ ПРИЗНАКОВ
X = np.hstack([x_encoded.toarray(), data[["weight", "height"]].values])  

# ЦЕЛЕВАЯ ПЕРЕМЕННАЯ
y = data["label"]

# ПОДГОТОВКА НОВОГО ОБЪЕКТА
new = encoder.transform([["dog"]]).toarray()                      # кодируем "dog"

# ДЕРЕВО РЕШЕНИЙ
tree = DecisionTreeClassifier(max_depth=2).fit(X, y)             # обучаем дерево (глубина 2)
new_data = np.hstack([new, [[2, 5]]])                            # добавляем вес 2 и рост 5
print(f"Decisison tree: {tree.predict([new_data])}\n")       
# KNN
knn = KNeighborsClassifier(n_neighbors=3)                       # создаём KNN с 3 соседями
knn.fit(X, y)                                                   # обучаем KNN

print(knn.predict(new_data))                                    # прогноз KNN