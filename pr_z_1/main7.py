
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Процесс создания мешка (Векторизация)
text = [
    "Плохо выглядишь",# негативное
    "Ты плохой",# негативное
    "Полный провал",# негативное
    "Отстой"# негативное
    "В тебе нет ничего хорошего" # негативное
    "У тебя ничего не получится" # негативное
    "Ты глупый" # негативное
    "Отстань" # негативное
    "Без обид" # негативное

    "У тебя получится" # позитивное
    "В тебе нет ничего плохого" # позитивное
    "Хорошего дня"# позитивное
    "Ты крутой"# позитивное
    "С тобой приятно общаться"# позитивное
    "Ты приятный человек",# позитивное
    "Спасибо",# позитивное
    "Ты хорошо выглядишь" # позитивное
    "Я тебя люблю" # позитивное
    "Обнимаю" # позитивное
    "Доброй ночи" # позитивное

    
]

lable = [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1]# 1- позитивное, 0 - негативное

text_train, text_test, y_train, y_test = train_test_split(text, lable, test_size=0.33, random_state=30)


pipe = make_pipeline (
    CountVectorizer(), 
    MultinomialNB()
)

# Обучение классификатора
pipe.fit(text_train, y_train)
y_pred = pipe.predict(text_test)
print(f"Accuracy:{accuracy_score(y_test,y_pred)}")

# Проверка на новых  сообщениях
print("New:", pipe.predict(["Отстойно выглядишь"])[0])
print("New:", pipe.predict(["Доброго дня"])[0])