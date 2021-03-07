import pandas as pd
from sklearn.tree import DecisionTreeClassifier

admission_data = pd.read_csv('data.csv').drop(columns=['Serial No'])

X = admission_data.drop(columns=['AdminChance'])
y = admission_data['AdminChance']

model = DecisionTreeClassifier()
model.fit(X,y)


#For Example if you get full on all exams
predictions = model.predict([[340,120,50,100,50,1000,1]])
predictions

#For Example if none marks in exams
predictions = model.predict([[0,0,0,0,0,0,0]])
predictions
