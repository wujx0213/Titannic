#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from DataHandle import featureHandle
from DataHandle import deleteProperty
from sklearn import tree
from sklearn import linear_model
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import Imputer
import sklearn.preprocessing as sp
from sklearn import svm
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier



train = pd.read_csv(r'E:\project\python\Titannic\data\train.csv')
test = pd.read_csv('E:\\project\\python\\Titannic\\data\\test.csv')
trainHandled = featureHandle(train)
trainName = ['SibSp','Parch','PassengerId','Ticket']
trainHandled = deleteProperty(trainHandled,trainName)
trainAgeHandled = trainHandled[trainHandled['Age'].notnull()]
trainNoAgeHandled = trainHandled[trainHandled['Age'].isnull()].drop('Age',1)
#print(trainNoAgeHandled)
testName = ['SibSp','Parch','Ticket']
testHandled = featureHandle(test)
print(len(testHandled))
testHandled = deleteProperty(testHandled,testName)
testAgeHandled = testHandled[testHandled['Age'].notnull()]
testNoAgeHandled = testHandled[testHandled['Age'].isnull()].drop('Age',1)
#testAgeHandled.to_csv('E:\\project\\python\\Titannic\\data\\testAge.csv')
#print(testNoAgeHandled)
#X = [[0.,1.],[1.,2.]]
#print(trainAgeHandled)
trainAgeHandledArray = trainAgeHandled.as_matrix()
trainNoAgeHandledArray = trainNoAgeHandled.as_matrix()
testAgeHandledArray = testAgeHandled.as_matrix()
testNoAgeHandledArray = testNoAgeHandled.as_matrix()
#print(trainAgeHandledArray)
'''
数据处理阶段结束，开始训练阶段
'''
X_Age=trainAgeHandledArray[:,1:9]
y_Age=trainAgeHandledArray[:,0]
y_Age = y_Age.astype(float)
#print(y_Age)
X_NoAge = trainNoAgeHandledArray[:,1:8]
y_NoAge = trainNoAgeHandledArray[:,0]
y_NoAge = y_NoAge.astype(float)
#sp.normalize(X_Age,norm = 'max',axis = 0)
#sp.normalize(X_NoAge,norm = 'max',axis = 0)
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(6, 3), random_state=1)
#clf = svm.SVC(kernel='rbf', gamma=0.7, C=1.0)
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)
'''
模型融合
'''
clf1 = RandomForestClassifier(max_depth=4, random_state=1,n_estimators = 9)
clf2= svm.SVC(kernel='rbf', gamma=0.7, C=1.0,probability=True)
clf3 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(6, 3), random_state=1)
clf4 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)
clf5 = tree.DecisionTreeClassifier()
clf6 = linear_model.LogisticRegression(C=1e5)
clf =  VotingClassifier(estimators=[('rf',clf1),('mlp', clf3),('dt',clf5),('lr',clf6)],voting='soft', weights=[8,1,1,1])

clf.fit(X_Age,y_Age)
joblib.dump(clf, 'E:\\project\\python\\Titannic\\data\\ageVoting.pkl')
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(6, 3), random_state=1)
clf.fit(X_NoAge,y_NoAge)
joblib.dump(clf, 'E:\\project\\python\\Titannic\\data\\noageVoting.pkl')
'''
训练阶段结束，测试阶段开始
'''
#print(testAgeHandledArray[0])
clf_age = joblib.load('E:\\project\\python\\Titannic\\data\\ageVoting.pkl')
clf_noage = joblib.load('E:\\project\\python\\Titannic\\data\\noageVoting.pkl')
X_Age = testAgeHandledArray[:,1:9]
#X_Age = X_Age.astype(float)
X_NoAge = testNoAgeHandledArray[:,1:8]
#print(len(X_Age))
X_Age = Imputer().fit_transform(X_Age)
X_NoAge = Imputer().fit_transform(X_NoAge)
#sp.normalize(X_Age,norm = 'max',axis = 0)
#sp.normalize(X_NoAge,norm = 'max',axis = 0)
#print(X_Age)
#X_NoAge = X_NoAge.astype(float)
'''
不使用投票得到预测


result_age = clf_age.predict(X_Age).astype(int)
result_noage = clf_noage.predict(X_NoAge).astype(int)
'''
'''
通过投票得到预测结果
'''
result_age_prob = clf_age.predict_proba(X_Age)
result_noage_prob = clf_noage.predict_proba(X_NoAge)
print(result_age_prob)
#print(len(result_age_prob)+len(result_noage_prob))
result_age = np.zeros( (len(result_age_prob),1), dtype=np.int16 )
result_noage = np.zeros( (len(result_noage_prob),1), dtype=np.int16 )
#print(result_age)

'''
通过投票的概率得出最后的判别结果
'''
for i in range(len(result_age_prob)):
    if (result_age_prob[i][1] > 0.5):
        result_age [i][0] = 1;
#print(result_age)
if(result_age_prob[4][1] > 0.5):
    print('aaaaa')
for i in range(len(result_noage_prob)):
    if (result_noage_prob[i][1] > 0.5):
        result_noage [i][0] = 1;
#print(len(result_age)+len(result_noage))
#print(result_noage_prob)

'''
#将narray转化为dataframe
ageDataframe = DataFrame(result_age,columns = ['label'])
noageDataframe = DataFrame(result_noage,columns = ['label'])
'''
#print(result_age)
'''
dataframe 中添加属性列
'''

testAgeHandled.insert(loc = 1,column = 'Survived',value=result_age)
testNoAgeHandled.insert(loc = 1,column = 'Survived',value=result_noage)
#print(len(testNoAgeHandled))
#print(testAgeHandled[['PassengerId','label']])
testIdLabel = testAgeHandled[['PassengerId','Survived']].append(testNoAgeHandled[['PassengerId','Survived']]).sort_values(by='PassengerId', ascending=True)

testIdLabel.to_csv('E:\\project\\python\\Titannic\\data\\result_voting.csv')
#print(result_noage)
#print(ageDataframe)
#print(testAgeLabel)



