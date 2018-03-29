from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
#import xgboost as xgb
from sklearn import cross_validation
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV

titanic = pd.read_csv('E:\\project\\python\\Titannic\\data_2\\train_handle.csv')
titanic_test = pd.read_csv('E:\\project\\python\\Titannic\\data_2\\test_handle.csv')

predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","Title","NameLength"]
alg = LinearRegression()
#print(titanic.shape[0])


'''
交叉验证过程
'''
kf = KFold(titanic.shape[0],n_folds=3,random_state=1)
predictions = []
for train,test in kf:
    train_predictors = (titanic[predictors].iloc[train,:])
    train_target = (titanic["Survived"]).iloc[train]
    alg.fit(train_predictors,train_target)
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)
#print(len(predictions[1])+len(predictions[0])+len(predictions[2]))

'''
计算精确度
逻辑回归
'''
predictions = np.concatenate(predictions,axis=0)
predictions[predictions > .2] = 1
predictions[predictions <= .2] = 0
accuracy = sum(predictions[predictions == titanic["Survived"]])/len(predictions)
#print(accuracy)


alg = LogisticRegression(random_state = 1)
scores = cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=3)
#print(scores.mean())
'''
随机森林
'''
'''
#param_test1 = {'n_estimators':list(range(10,101,5)),'random_state':list(range(0,50,2)),'min_samples_split':list(range(2,50,2)),'min_samples_leaf':list(range(1,50,2)),'max_depth':list(range(1,20,2))}
#param_test1 = {'n_estimators':list(range(10,101,5))}#30
#param_test1 = {'random_state':list(range(0,50,2))}#18
#param_test1 = {'max_depth':list(range(1,50,2))}#3
#param_test1 = {'min_samples_leaf':list(range(1,50,2))}#9
#param_test1 = {'min_samples_split':list(range(2,100,2))}#36
param_test1 = {'max_features':['auto','sqrt','log2',None]}#36
#print(type(param_test1),param_test1.values())
#gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10),param_grid = param_test1, scoring='roc_auc',cv=5)
#gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10),param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(max_features='auto' ,n_estimators=30,random_state=18,max_depth=3,min_samples_leaf=9,min_samples_split=36),param_grid = param_test1, scoring='roc_auc',cv=5)

print('nihaonihaoniahoanihao')
gsearch1.fit(titanic[predictors],titanic["Survived"])
#print(gsearch1.grid_scores_)
#print(gsearch1.best_params_, gsearch1.best_score_)
'''

alg = RandomForestClassifier(max_features='auto' ,n_estimators=30,random_state=18,max_depth=3,min_samples_leaf=9,min_samples_split=36)
kf = cross_validation.KFold(titanic.shape[0],n_folds=8,random_state=2)
scores = cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv = kf)
print(scores.mean())
''''''
alg.fit(titanic[predictors],titanic["Survived"])
joblib.dump(alg,'E:\\project\\python\\Titannic\\data_2\\RandomForestClassifier.pkl')
importances = alg.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(titanic[predictors].shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, predictors[f], importances[indices[f]]))
'''
GradientBoostingClassifier
'''
'''
#param_test1 = {'n_estimators':list(range(10,101,5)),'random_state':list(range(0,50,2)),'min_samples_split':list(range(2,50,2)),'min_samples_leaf':list(range(1,50,2)),'max_depth':list(range(1,20,2))}
#param_test1 = {'n_estimators':list(range(10,200,5))}#130
#param_test1 = {'random_state':list(range(0,50,2))}#24
#param_test1 = {'max_depth':list(range(1,50,2))}#3
#param_test1 = {'min_samples_leaf':list(range(1,50,2))}#9
#param_test1 = {'min_samples_split':list(range(2,100,2))}#36
#param_test1 = {'max_features':['auto','sqrt','log2',None]}#36
#param_test1 = {'max_leaf_nodes':list(range(2,50,2))}#6
#print(type(param_test1),param_test1.values())
#gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10),param_grid = param_test1, scoring='roc_auc',cv=5)
#gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10),param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators=130,max_leaf_nodes=6,max_depth=5,random_state=24,min_samples_split=40),param_grid = param_test1, scoring='roc_auc',cv=5)

print('nihaonihaoniahoanihao')
gsearch1.fit(titanic[predictors],titanic["Survived"])
print(gsearch1.grid_scores_)
print(gsearch1.best_params_, gsearch1.best_score_)
'''



alg = GradientBoostingClassifier(n_estimators=130,max_leaf_nodes=6,max_depth=5,random_state=24,min_samples_split=40)
kf = cross_validation.KFold(titanic.shape[0],n_folds=9,random_state=2)
scores = cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv = kf)
print(scores.mean())
alg.fit(titanic[predictors],titanic["Survived"])
joblib.dump(alg,'E:\\project\\python\\Titannic\\data_2\\GradientBoostingClassifier.pkl')
'''
xgBoost

dtrain=xgb.DMatrix(titanic[predictors],label=titanic["Survived"])
dtest=xgb.DMatrix(titanic_test)

params={'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':4,
    'lambda':10,
    'subsample':0.75,
    'colsample_bytree':0.75,
    'min_child_weight':2,
    'eta': 0.025,
    'seed':0,
    'nthread':8,
     'silent':1}

watchlist = [(dtrain,'train')]
bst=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)

ypred=bst.predict(dtest)
print(ypred)
'''
'''
特征选择
'''
selector = SelectKBest(f_classif,k=5)
selector.fit(titanic[predictors],titanic["Survived"])
scores = -np.log10(selector.pvalues_)
plt.bar(range(len(predictors)),scores)
plt.xticks(range(len(predictors)),predictors,rotation = 'vertical')
#plt.show()

#predictors = ["Pclass","Sex","Fare","Title","NameLength"]
predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize"]
'''
#param_test1 = {'n_estimators':list(range(10,101,5)),'random_state':list(range(0,50,2)),'min_samples_split':list(range(2,50,2)),'min_samples_leaf':list(range(1,50,2)),'max_depth':list(range(1,20,2))}
#param_test1 = {'n_estimators':list(range(10,300,5))}#250
#param_test1 = {'random_state':list(range(0,50,2))}#18
#param_test1 = {'max_depth':list(range(1,50,2))}#3
#param_test1 = {'min_samples_leaf':list(range(1,50,2))}#9
#param_test1 = {'min_samples_split':list(range(2,100,2))}#36
#param_test1 = {'max_features':['auto','sqrt','log2',None]}#36
#print(type(param_test1),param_test1.values())
#gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10),param_grid = param_test1, scoring='roc_auc',cv=5)
#gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10),param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=250,random_state=18,max_depth=3,min_samples_leaf=1,min_samples_split=36),param_grid = param_test1, scoring='roc_auc',cv=5)

print('nihaonihaoniahoanihao')
gsearch1.fit(titanic[predictors],titanic["Survived"])
print(gsearch1.grid_scores_)
print(gsearch1.best_params_, gsearch1.best_score_)
'''
#alg = RandomForestClassifier(random_state=2,n_estimators=9,min_samples_split=4,min_samples_leaf=2)
alg = RandomForestClassifier(n_estimators=250,random_state=18,max_depth=3,min_samples_leaf=1,min_samples_split=36)
kf = cross_validation.KFold(titanic.shape[0],n_folds=30,random_state=3)
scores = cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv = kf)
print(scores.mean())
''''''
alg.fit(titanic[predictors],titanic["Survived"])
joblib.dump(alg, 'E:\\project\\python\\Titannic\\data_2\\RandomForestClassifier_2.pkl')
importances = alg.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(titanic[predictors].shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, predictors[f], importances[indices[f]]))




alg = GradientBoostingClassifier(n_estimators=490,max_leaf_nodes=4,max_depth=2,random_state=6,min_samples_split=5)
kf = cross_validation.KFold(titanic.shape[0],n_folds=5,random_state=1)
scores = cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv = kf)
#print(scores.mean())
alg.fit(titanic[predictors],titanic["Survived"])
joblib.dump(alg,'E:\\project\\python\\Titannic\\data_2\\GradientBoostingClassifier_2.pkl')



