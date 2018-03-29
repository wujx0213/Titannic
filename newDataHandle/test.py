import pandas as pd
from sklearn.externals import joblib
#from sklearn.ensemble import RandomForestClassifier
titanic_test = pd.read_csv('E:\\project\\python\\Titannic\\data_2\\test_handle.csv')
clf = joblib.load('E:\\project\\python\\Titannic\\data_2\\RandomForestClassifier_2.pkl')
#predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","Title","NameLength"]
predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize"]
#predictors = ["Pclass","Sex","Fare","Title","NameLength"]
result = clf.predict(titanic_test[predictors])
titanic_test["Survived"] = result
titanic_test.to_csv(('E:\\project\\python\\Titannic\\data_2\\result.csv'))
