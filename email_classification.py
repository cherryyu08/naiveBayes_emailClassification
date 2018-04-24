from text_classification import *


def textParse(bigString):  # input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)  # 分隔符是除单词、数字之外的任意字符串
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


# 词袋模型，比文本分类中的词集模型的好处是，当遇到一个单词，会增加词向量中的对应值，而不是只将对应的数值设为1
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 对贝叶斯垃圾邮件分类器进行自动化处理
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        # print wordList
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 词向量
    vocabList = createVocabList(docList)  # create vocabulary
    trainingSet = range(50)
    testSet = []  # create test set
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount) / len(testSet))
    # return vocabList,fullText
