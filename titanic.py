import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt



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

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1.0, 20), verbose=0, plot=True):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("number of training dataset")
        plt.ylabel("score")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"training dataset score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"cross validation score")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

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

plot_learning_curve(clf, "learning curve", x, y)