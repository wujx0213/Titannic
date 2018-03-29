import pandas as pd
import re
titanic = pd.read_csv('E:\\project\\python\\Titannic\\data_2\\train.csv')
titanic_test = pd.read_csv('E:\\project\\python\\Titannic\\data_2\\test.csv')
#print(titanic.head(5))
#print(titanic.describe())
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
#print(titanic_test.describe())
titanic.loc[titanic["Sex"] =="male","Sex"] = 0
titanic.loc[titanic["Sex"] =="female","Sex"] = 1
titanic_test.loc[titanic_test["Sex"] =="male","Sex"] = 0
titanic_test.loc[titanic_test["Sex"] =="female","Sex"] = 1
'''
输出该列所有属性值
'''
#print(titanic["Embarked"].unique())
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S","Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C","Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q","Embarked"] = 2
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"] == "S","Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C","Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q","Embarked"] = 2
#print(titanic["Embarked"].unique())
#print(titanic.head(5))
#print(titanic["Sex"].unique())
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
titanic["NameLength"] = titanic["Name"].apply(lambda x :len(x))
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x :len(x))
'''
将单词分类统计
'''
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.',name)
    if(title_search):
        return title_search.group(1)
    return

title = titanic["Name"].apply(get_title)
title_test = titanic_test["Name"].apply(get_title)
print(pd.value_counts(title_test))
'''
将对应得单词用数字替换
'''
title_mapping = {"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Dr":5,"Rev":6,"Major":7,"Col":8,"Mlle":9,"Countess":10,"Ms":11,"Lady":12,"Jonkheer":13,"Don":14,"Mme":15,"Capt":16,"Sir":17,"Dona":18}
for k,v in title_mapping.items():
    title[title==k] = v
    title_test[title_test==k] = v
print(pd.value_counts(title_test))
print(title_test)
titanic["Title"] = title
titanic_test["Title"] = title_test
#print(titanic.head(5))
#titanic.to_csv('E:\\project\\python\\Titannic\\data_2\\train_handle.csv')
titanic_test.to_csv('E:\\project\\python\\Titannic\\data_2\\test_handle.csv')
