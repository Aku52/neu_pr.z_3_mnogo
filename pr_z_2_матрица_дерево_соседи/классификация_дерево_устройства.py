#"Классификация типа устройства (стандартный/премиум) на основе категории устройства,
#  бренда и цены с помощью дерева решений и KNN"
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data = pd.DataFrame({
    "devices":["phone","comp","comp","notebook","phone","notebook","tablet","tablet"],
    "brand":["samsung","LG","apple","honor","apple","huawei","samsumg","redmi"],
    "price":[1200,4500,12000,4000,3600,2350,4000,2000],
    "type": [ "standart","standart","premium","premium","premium","standart","premium","standart"]
})

encoder= OneHotEncoder()
x_encoded = encoder.fit_transform(data[["devices","brand"]])

X = np.hstack([x_encoded.toarray(),data[["price"]].values])

y = data["type"]
new= encoder.transform([["comp","LG"]]).toarray()


tree= DecisionTreeClassifier(max_depth=2).fit(X,y)
new_data=np.hstack([new, [[5000]]])
print(f"Decisison tree: {tree.predict(new_data)}\n")


knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X,y)

print (knn.predict(new_data))