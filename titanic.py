import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


def set_numeralization(data, labelArray, labelEncoderArray):
    for label in labelArray:
        labelEncoder = preprocessing.LabelEncoder()
        labelEncoder.fit(data[label].unique())

        labelEncoderArray.append(labelEncoder)
        data[label] = labelEncoder.transform(data[label])

    return data
        
def printLabelMapping(labelEncoderArr):
    for labelEncoder in labelEncoderArr:
        labelMapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
        print(labelMapping)

def set_missing_age(df):
    # 把数值类型特征取出来，放入随机森林中进行训练
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    # 乘客分成已知年龄和未知年龄两个部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    # 目标数据y
    y = known_age[:,0]
    # 特征属性数据x
    x = known_age[:,1:]

    # 利用随机森林进行拟合
    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(x,y)
    # 利用训练的模型进行预测
    predictedAges = rfr.predict(unknown_age[:,1::])
    # 填补缺失的原始数据
    df.loc[(df.Age.isnull()),'Age'] = predictedAges
    return df

train = pd.read_csv('titanic/train.csv')
test = pd.read_csv('titanic/test.csv')

print(test.isnull().sum())
print(train.isnull().sum())

labelEncoderArray = []
train = set_numeralization(train, ['Pclass', 'Sex', 'Embarked'], labelEncoderArray)
printLabelMapping(labelEncoderArray)


train_df = train.filter(regex='Survived|Age|SibSp|Parch|Fare|Embarked|Sex|Pclass')
train_df = set_missing_age(train_df)
train_df.isnull().sum()

clf = linear_model.LogisticRegression(solver='liblinear',C=1.0,penalty='l2',tol=1e-6)

train_np = train_df.values
print(train_np)
# 获取y
y = train_np[:,0]
# 获取自变量x
x = train_np[:,1:]
print(x, y)

clf.fit(x,y)

score = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
print(score)
print(score.mean())