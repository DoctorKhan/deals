""" Classification

The objective of this task is to build a classifier that can tell us whether a new, unseen deal 
requires a coupon code or not. 

We would like to see a couple of steps:

1. You should use bad_deals.txt and good_deals.txt as training data
2. You should use test_deals.txt to test your code
3. Each time you tune your code, please commit that change so we can see your tuning over time

Also, provide comments on:
    - How general is your classifier?
    - How did you test your classifier?

"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from gensim import matutils, corpora, models, similarities
#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
#                    level=logging.INFO)

class Task3:
    def __init__(self):
        """Read good deals in"""
        pGood = '../data/good_deals.txt'
        msGood = []
        vbGood = []
        for line in open(pGood, 'r'):
            msGood.append(line.strip().split())
            vbGood.append(1)

        """Read bad deals in"""
        pBad  = '../data/bad_deals.txt'
        msBad  = []
        vbBad = []        
        for line in open(pBad, 'r'):
            msBad.append(line.strip().split())
            vbBad.append(0)

        """Read test deals in"""
        pTest = '../data/test_deals.txt'
        msTest = []
        for line in open(pTest, 'r'):
            msTest.append(line.strip().split())
    
        """Split the data into training and cross-validation sets"""
        msTrain = msGood[10:30] + msBad[10:30]
        msCross = msGood[0:10]  + msBad[0:10]
        vbTrain = vbGood[10:30] + vbBad[10:30]
        vbCross = vbGood[0:10]  + vbBad[0:10]

        oDict = corpora.Dictionary(msTrain)
        nTerms = len(oDict)
        oCorpTrain = [oDict.doc2bow(text) for text in msTrain]
        oCorpTest = [oDict.doc2bow(text) for text in msTest]

        """Fit SVC model"""
        nDocs = len(vbTrain)
        mCorpusTrain = matutils.corpus2dense(oCorpTrain, nTerms, nDocs).T

        """This value of C is just high enough to properly predict 30% of 
        the cross-validation set properly and not so low that it doesn't
        predict the training set properly. 30% is apparently the best it can
        do with the cross-validation set. More training data would be helpful.

        I.e. it is general enough to fit the cross-validation set as best
        as it will be fit by change C."""
        oSVCModel = svm.SVC(C=35)
        oSVCModel.fit(mCorpusTrain, vbTrain) 
        vbPredTrain = oSVCModel.predict(mCorpusTrain)

        """Tess classifier on cross-validation set"""
        oCorpCV = [oDict.doc2bow(text) for text in msCross]
        nCrossDocs = len(vbCross);
        mCorpCV = matutils.corpus2dense(oCorpCV, nTerms, nCrossDocs).T
        vbPredCross = oSVCModel.predict(mCorpCV)
                    
        """Use SVC model to make prediction."""
        nTestDocs = len(list(oCorpTest))
        mTest = matutils.corpus2dense(oCorpTest, nTerms, nTestDocs).T
        vbPredTest = oSVCModel.predict(mTest)
        
        """Store member variables for retrieval"""
        self.vbPredTest = vbPredTest
        self.vbPredCross = vbPredCross
        self.vbPredTrain = vbPredTrain