import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np

data = pd.DataFrame({
    "fruit":["apple","orange","banan","pineapple","orange","apple"],
    "weight":[100,80,50,90,125,150],
    "label": [ "low","low","average","low","hight","hight"]
})

encoder= OneHotEncoder()
x_encoded = encoder.fit_transform(data[["fruit"]])

x = np.hstack([x_encoded.toarray(),data[["weight"]].values])

y = data["label"]
clf =DecisionTreeClassifier()
clf.fit(x,y)

new = encoder.transform(["apple"]).toarray()
new_data = np.hstack([new,[[115]]])

print (clf.predict(new_data))