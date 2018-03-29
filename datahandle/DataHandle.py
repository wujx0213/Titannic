#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
def featureHandle(data):
    data['Family'] = data.apply(lambda x:0,axis=1 );
    for i  in range(len(data['Name'])):
        nameString = data['Name'][i];
        sexString = data['Sex'][i];
        cabinString = data['Cabin'][i];
        embarkedString = data['Embarked'][i];
        sibSPInt = data['SibSp'][i];
        parchInt = data['Parch'][i];
        identify = nameString.split(',')[1].split('.')[0].replace(' ','');
        #print(sibSPInt+parchInt);
        #对“name"这一属性列进行处理
        if(identify == 'Mr'):
           # data['Name'][i]='1'; #修改DataFrame中的某一个值是不能使用这种方式的，只能使用下面这种方法
            data.loc[i, 'Name'] =1;
        if(identify=='Mrs'):
            data.loc[i, 'Name'] =2;
        if(identify=='Miss'):
            data.loc[i, 'Name'] =3;
        if (identify == 'Master'):
            data.loc[i, 'Name'] = 4;
        if (identify == 'Dr'):
            data.loc[i, 'Name'] = 5;
        if(identify!='Mr' and identify!='Mrs' and identify!='Miss' and identify!='Master' and identify!='Dr'):
            data.loc[i, 'Name'] = 0;
        #对"Sex"这一属性列进行处理
        if(sexString == 'female'):
            data.loc[i, 'Sex'] = 1;
        if (sexString == 'male'):
            data.loc[i, 'Sex'] = 0;
        #对“Cabin”这一属性列进行处理
        if(type(cabinString)== float):
            data.loc[i, 'Cabin'] = 0;
        if(type(cabinString)== str):
            if(cabinString[0]=='A'):
                data.loc[i, 'Cabin'] = 1;
            if(cabinString[0]=='B'):
                data.loc[i, 'Cabin'] = 2;
            if (cabinString[0] == 'C'):
                data.loc[i, 'Cabin'] = 3;
            if (cabinString[0] == 'D'):
                data.loc[i, 'Cabin'] = 4;
            if (cabinString[0] == 'E'):
                data.loc[i, 'Cabin'] = 5;
            if (cabinString[0] == 'F'):
                data.loc[i, 'Cabin'] = 6;
            if (cabinString[0] == 'G'):
                data.loc[i, 'Cabin'] = 7;
            if (cabinString[0] == 'T'):
                data.loc[i, 'Cabin'] = 8;
        #对embarked属性列进行处理
        if(embarkedString=='C'):
            data.loc[i, 'Embarked'] = 1;
        if (embarkedString == 'Q'):
            data.loc[i, 'Embarked'] = 2;
        if (embarkedString == 'S'):
            data.loc[i, 'Embarked'] = 3;
        if(type(embarkedString)== float):
            data.loc[i, 'Embarked'] = 0;
        #对SibSp和Parch进行加和够着family属性
        data.loc[i,'Family'] = sibSPInt+parchInt;
    return data;
def deleteProperty(data,name):
    for i in name:
        data = data.drop(i,1);
    return data;
#print(trainHandled);
#print(testHandled);

