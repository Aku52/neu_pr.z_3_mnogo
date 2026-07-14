#"Напишите программу, которая создаёт DataFrame с категориальным признаком 'fruit' и числовым признаком 'weight'-вес, 
# а затем преобразует категориальный признак в числовой формат с помощью OneHotEncoder. 
# Выведите исходные данные, закодированную матрицу и названия новых признаков."

# ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# СОЗДАНИЕ ДАННЫХ
data = pd.DataFrame({
    "fruit": ["apple", "orange", "banan", "pineapple", "orange", "apple"],  # категориальный признак
    "weight": [100, 80, 50, 90, 125, 150]                                   # числовой признак
})
print("Исходные данные:")
print(data)

# ONE-HOT ENCODING
encoder = OneHotEncoder() # создаём кодировщик
encoded = encoder.fit_transform(data[["fruit"]])   # преобразуем фрукты в бинарные столбцы

print("\nЗакодированные данные (матрица):")
print(encoded.toarray()) # вывод матрицы (0 и 1)

print("\nНазвания новых признаков:")
print(encoder.get_feature_names_out(["fruit"])) # названия столбцов после кодирования