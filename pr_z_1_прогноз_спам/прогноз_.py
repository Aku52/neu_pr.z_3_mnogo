# ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ДАННЫЕ: продажи по неделям (с пропусками и выбросами)
sales = np.array([100, 0, 120, 0, 0, 500, 450])

# ФУНКЦИЯ СКОЛЬЗЯЩЕГО СРЕДНЕГО
def moving_average(series, window=3):
    if len(series) < window:           # если данных меньше окна
        return np.mean(series)         # среднее по всем данным
    return np.mean(series[-window:])   # среднее последних window значений (исправлено)

# Прогноз скользящим средним
ma_pred = moving_average(sales, window=3)
print(sales[-3])  # последние 3 значения продаж

# ЛИНЕЙНАЯ РЕГРЕССИЯ
x = np.arange(len(sales)).reshape(-1, 1)  # номера недель (0,1,2,...)
y = sales                                 # продажи

model = LinearRegression()
model.fit(x, y)                           # обучение модели

next_week = np.array([[len(sales)]])      # следующая неделя (№7)
linear_pred = model.predict(next_week)[0] # прогноз на следующую неделю
print(f"Linear regression prediction next week: {linear_pred}")