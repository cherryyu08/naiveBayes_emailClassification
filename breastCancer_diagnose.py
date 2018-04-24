import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

ws_data = pd.read_csv("data.csv")
# 提取标签
diagnoses = ws_data.diagnosis
# 提取后面30列作为特征
predict_features = ws_data.iloc[:, 2:32]


# a为准确率
# a = cross_val_score(nbClf, X=predict_features, y=diagnoses, cv=5, scoring='accuracy')


# 使用不同的分类器训练
class winsconBC1:
    def __init__(self, data_x, data_y, method, rand_seed=42, Nfolds=3, shuffled=True):
        from sklearn.model_selection import StratifiedKFold
        self.data_x = data_x
        self.data_y = data_y
        self.clf = method
        self.Rseed = rand_seed
        self.Nfolds = Nfolds
        self.shuffle = shuffled
        self.classifier = method
        # StratifiedKFold能把数据集按照良恶性一定比例分成n_fold份
        # 做五折交叉验证，四分作为训练集，一分作为测试集
        # 把几种分类算法在测试集上评估预测的准确率
        self.sfk = StratifiedKFold(n_splits=self.Nfolds, random_state=self.Rseed, shuffle=self.shuffle)

    def classify(self):
        accuracy = list()
        for TR, TS in self.sfk.split(self.data_x, self.data_y):
            train_x, train_y = predict_features.iloc[TR, :], diagnoses[TR]
            test_x, test_y = predict_features.iloc[TS, :], diagnoses[TS]

            if self.classifier == 'logistic':
                from sklearn.linear_model import LogisticRegression
                logClf = LogisticRegression(random_state=42)
                logClf.fit(train_x, train_y)
                accuracy.append(logClf.score(X=test_x, y=test_y))

            if self.classifier == 'SGD':
                from sklearn.linear_model import SGDClassifier
                SGDClf = SGDClassifier(random_state=self.Rseed)
                SGDClf.fit(X=train_x, y=train_y)
                accuracy.append(SGDClf.score(X=test_x, y=test_y))

            if self.classifier == 'SVM':
                from sklearn.svm import SVC
                svmClf = SVC(random_state=self.Rseed)
                svmClf.fit(X=train_x, y=train_y)
                accuracy.append(svmClf.score(X=test_x, y=test_y))

            # 随即森林
            if self.classifier == 'randomforest':
                from sklearn.ensemble import RandomForestClassifier
                rmClf = RandomForestClassifier()
                rmClf.fit(X=train_x, y=train_y)
                accuracy.append(rmClf.score(X=test_x, y=test_y))

        return np.array(accuracy)


# 改进
# 手动设置了一些参数，改进之后的模型有了明显提升
class winsconBC2:
    def __init__(self, data_x, data_y, method, rand_seed=42, Nfolds=3, shuffled=True):
        from sklearn.model_selection import StratifiedKFold
        self.data_x = data_x
        self.data_y = data_y
        self.clf = method
        self.Rseed = rand_seed
        self.Nfolds = Nfolds
        self.shuffle = shuffled
        self.classifier = method
        # StratifiedKFold能把数据集按照良恶性一定比例分成n_fold份
        # 做五折交叉验证，四分作为训练集，一分作为测试集
        # 把几种分类算法在测试集上评估预测的准确率
        self.sfk = StratifiedKFold(n_splits=self.Nfolds, random_state=self.Rseed, shuffle=self.shuffle)

    def classify(self):
        accuracy = list()
        for TR, TS in self.sfk.split(self.data_x, self.data_y):
            train_x, train_y = predict_features.iloc[TR, :], diagnoses[TR]
            test_x, test_y = predict_features.iloc[TS, :], diagnoses[TS]

            if self.classifier == 'logistic':
                from sklearn.linear_model import LogisticRegression
                logClf = LogisticRegression()
                logClf.fit(train_x, train_y)
                accuracy.append(logClf.score(X=test_x, y=test_y))

            if self.classifier == 'SGD':
                from sklearn.linear_model import SGDClassifier
                # 手动设置参数
                SGDClf = SGDClassifier(loss='log', max_iter=1000, learning_rate='optimal')
                SGDClf.fit(X=train_x, y=train_y)
                accuracy.append(SGDClf.score(X=test_x, y=test_y))

            if self.classifier == 'SVM':
                from sklearn.svm import SVC
                # kernel默认是rbf
                svmClf = SVC(kernel='linear', C=1)
                svmClf.fit(X=train_x, y=train_y)
                accuracy.append(svmClf.score(X=test_x, y=test_y))

            # 随机森林
            if self.classifier == 'randomforest':
                from sklearn.ensemble import RandomForestClassifier
                rmClf = RandomForestClassifier()
                rmClf.fit(X=train_x, y=train_y)
                accuracy.append(rmClf.score(X=test_x, y=test_y))

            # 朴素贝叶斯
            if self.classifier == 'naivebayes':
                from sklearn.naive_bayes import GaussianNB
                nbClf = GaussianNB()
                nbClf.fit(X=train_x, y=train_y)
                accuracy.append(nbClf.score(X=test_x, y=test_y))

        return np.array(accuracy)


plt.figure()
plt.bar(np.arange(5), np.array([np.mean(winsconBC2(data_x=predict_features,
                                                  data_y=diagnoses, method=i, Nfolds=5).classify())
                                for i in ['SGD', 'logistic', 'SVM', 'randomforest', 'naivebayes']]))
plt.xticks(np.arange(5), ('SGD', 'logistic', 'SVM', 'randomforest', 'naivebayes'))
plt.axhline(0.95, c='r')  # 加一条accuracy=0.95的基准线做参考
plt.show()

# 需要注意， sklearn 里面的朴素贝叶斯分类器predict_proba的函数预测的概率并不准确
