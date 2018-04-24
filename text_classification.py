import numpy as np


# 文本分类数据集
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]    # 1 is abusive, 0 not

    return postingList, classVec


# 创建词汇表（利用集合结构内元素的唯一性，创建一个包含所有词汇的词表。）
def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        # 去掉重复的部分
        vocabSet = vocabSet | set(document)  # union of the two sets

    return list(vocabSet)


'''
eq，测试文本1： ['love', 'my', 'dalmation']

　　　　词汇表：['cute', 'love', 'help', 'garbage', 'quit', 'I', 'problems', 'is', 'park', 'stop', 'flea', 'dalmation', 'licks', 'food', 'not', 'him', 'buying', 'posting', 'has', 'worthless', 'ate', 'to', 'maybe', 'please', 'dog', 'how', 'stupid', 'so', 'take', 'mr', 'steak', 'my']

　　　　向量化结果：[0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
'''


# 把输入文本根据词表转化为计算机可处理的01向量形式
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)  # [0, 0, 0, .....]
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


'''
在训练样本中计算先验概率 p(Ci) 和 条件概率 p(x,y | Ci)，本实例有0和1两个类别，所以返回p(x,y | 0)，p(x,y | 1)和p(Ci)。

　　此处有两个改进的地方：

　　　　（1）若有的类别没有出现，其概率就是0，会十分影响分类器的性能。所以采取各类别默认1次累加，总类别（两类）次数2，这样不影响相对大小。

　　　　（2）若很小是数字相乘，则结果会更小，再四舍五入存在误差，而且会造成下溢出。采取取log，乘法变为加法，并且相对大小趋势不变。
'''


# 训练模型
def trainNB0(trainMatrix, trainCategory):
    # 单词矩阵
    numTrainDocs = len(trainMatrix)
    # 单词标签
    numWords = len(trainMatrix[0])

    # 滥用词的概率（先验概率）
    pAbusive = sum(trainCategory)/float(numTrainDocs)

    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)      # change to ones()
    p0Denom = 2.0
    p1Denom = 2.0              # change to 2.0
    # 遍历矩阵i = 0,1,2...
    for i in range(numTrainDocs):
        # 滥用词
        if trainCategory[i] == 1:
            # 单词向量相加
            p1Num += trainMatrix[i]
            # 值相加
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # 计算条件概率
    p1Vect = np.log(p1Num/p1Denom)          # change to log()
    p0Vect = np.log(p0Num/p0Denom)          # change to log()

    return p0Vect, p1Vect, pAbusive


# 分类：根据计算后，哪个类别的概率大，则属于哪个类别
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 测试函数
# 加载数据集+提炼词表；
# 训练模型：根据六条训练集计算先验概率和条件概率；
# 测试模型：对训练两条测试文本进行分类。
def testingNB():
    # 返回单词列表和分类列表
    listOPosts, listClasses = loadDataSet()
    # 创建一个包含所有词汇的词表
    myVocabList = createVocabList(listOPosts)

    trainMat = []
    # 把单词转换成0和1的形式
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


testingNB()


# 这里做的是向量操作，可以吧向量操作看成是同时对向量里面的所有值同时进行单值操纵，看成单值操作，就比较好理解
# 测试了一下，可以说这个预测效果不是很准
# p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)不是很理解
# 缺点：词表只能记录词汇是否出现，不能体现这个词汇出现的次数。改进方法：采用词袋模型，见下面垃圾邮件分类实战。
