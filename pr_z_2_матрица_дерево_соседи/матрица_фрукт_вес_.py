#"Напишите программу, которая создаёт DataFrame с категориальным признаком 'fruit' и числовым признаком 'weight'-вес, 
# а затем преобразует категориальный признак в числовой формат с помощью OneHotEncoder. 
# Выведите исходные данные, закодированную матрицу и названия новых признаков."

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = pd.DataFrame({
    "fruit":["apple","orange","banan","pineapple","orange","apple"],
    "weight":[100,80,50,90,125,150]
})
print(data)
encoder = OneHotEncoder()
encoded = encoder.fit_transform(data[["fruit"]])

print(encoded.toarray())

print(encoder.get_feature_names_out(["fruit"]))