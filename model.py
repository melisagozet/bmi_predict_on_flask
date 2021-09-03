
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

bmi_prediction = pd.read_csv("dataset/500_Person_Gender_Height_Weight_Index.csv")


X = bmi_prediction[['Height', 'Weight']]
y = bmi_prediction['Index']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=3)

lm = LogisticRegression(solver='liblinear')
lm.fit(X_train, y_train)
print(lm.predict(X_test))
pickle.dump(lm, open('model.pkl', 'wb'))