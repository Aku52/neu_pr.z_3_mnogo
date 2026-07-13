#Классификация животных по весу и росту с помощью метода k-ближайших соседей (KNN)
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data = pd.DataFrame({
    "animal":["cat","cat","dog","dog","cat","dog"],
    "weight":[1,1.5,50,90,5,30],
    "height":[10,15.50,45,5,80],
    "label": [ "home","home","service","service","home","service"]
})

encoder= OneHotEncoder()
x_encoded = encoder.fit_transform(data[["animal"]])

x = np.hstack([x_encoded.toarray(),data[["weight","heigh"]].values])

y = data["label"]

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x,y)

new = encoder.transform(["dog"]).toarray()
new_data = np.hstack([new,[[47,28]]])

print (knn.predict(new_data))