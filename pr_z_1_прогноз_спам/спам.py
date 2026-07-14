
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

text = [
    "Купи айфон за 100 рублей",# негативное
    "Ты плохой",# негативное
    " Вы выиграли миллион, пришлите денег",# spam
    "Отчет: продажи за квартал 2025",# not spam
    "Cрочно! Позвони сейчас и получи приз",# spam
    "Напоминаем про оплату счета"# not spam
]

lable = [1,0,1,0,1,0]# 1- spam, 0 - not spam

# РАЗДЕЛЕНИЕ ДАННЫХ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
text_train, text_test, y_train, y_test = train_test_split(text, lable, test_size=0.33, random_state=42)

# ПАЙПЛАЙН: ВЕКТОРИЗАЦИЯ ТЕКСТА + КЛАССИФИКАЦИЯ
pipe = make_pipeline(
    CountVectorizer(),  # преобразует текст в "мешок слов" (числовые признаки)
    MultinomialNB()     # классификатор Наивный Байес
)

# ОБУЧЕНИЕ МОДЕЛИ
pipe.fit(text_train, y_train)               # модель запоминает связь текстов с метками
y_pred = pipe.predict(text_test)            # предсказание на тестовой выборке
print(f"Accuracy:{accuracy_score(y_test, y_pred)}")  # точность модели

# ПРОВЕРКА НА НОВЫХ СООБЩЕНИЯХ
print("New:", pipe.predict(["Поздравляем! Вы победитель. Пришлите нам свой CVV код"])[0])
print("New:", pipe.predict(["Добрый вечер! Встреча завтра в 12"])[0])