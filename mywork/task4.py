"""
Task 4:

In this task we present a real life dataset in the form of a supervised classification problem.
Ref: data/coupon_clickstream.csv

This dataset contains 50 observations and one target variable.
What we are trying to predict here is that given these 50 metrics how likely is a user to click on a coupon.

Your task is the following:

Perform exploratory analysis on the dataset to answer following questions:

1. Are there any redundant metrics in the 50 variables we have collected?
2. Are there any correlated metrics?
3. Will dimensionality reduction need to be applied?

Once you know what you are looking at perform the following tasks:

1. Find the optimal number of features that maximize the accuracy of predicting 'coupon_click'
2. Once you identify optimal number of features can you rank the features from most important to least?

Use optimal features you found above with different classifiers and evaulate your classifiers as to how general they are with relevant metrics.

"""

#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
#                    level=logging.INFO)
                    
import csv, numpy
from sklearn import svm
from sklearn.feature_selection import SelectPercentile, f_classif

class Task4:
    def __init__(self):

        pFile = "../data/coupon_clickstream.csv"

        """Read in data."""
        mAllData = numpy.genfromtxt(pFile, delimiter=',', skip_header=1)
        mX = mAllData[:, 0:50]
        vY = mAllData[:, 50]

        """Mean normalization and feature scaling"""
        vXMean = mX.mean(axis=0)
        mXMeanNorm = mX - vXMean
        mXScaledNorm = mXMeanNorm.std(axis = 0)
        mXNormScale = mXMeanNorm/mXScaledNorm

        """Identify redundancies"""
        mCovariance = numpy.corrcoef(mXNormScale.T)
        vvSigEig = (mCovariance>0.99).nonzero()
        n = len(vvSigEig[0])
        viRemove = [vvSigEig[0][ii] for ii in range(0,n-1) if vvSigEig[0][ii] < vvSigEig[1][ii]]

        """Remove redundancies"""
        mXRemoved = numpy.delete(mXNormScale, viRemove, axis = 1)

        """Feature selection"""

        """Other than the SVM weights, F-Test may be used to rank features
        as well as sparse model methods, but these may reduce accuracy."""

        """Univariate feature selection with F-test for feature scoring"""
        oSelector = SelectPercentile(f_classif, percentile=90)
        oSelector.fit(mXRemoved, vY)
        vScores = -numpy.log10(oSelector.pvalues_)
        vScores /= vScores.max()
        """This indicates the relative importance of the metrics in
        where the noisier terms occur first"""
        vFScoreRanks = vScores.argsort()
        """Removing the first 10 features improves cross-validation success
        by half a percent"""
        mXRemoved = numpy.delete(mXRemoved, vFScoreRanks[0:10], axis = 1)

        """Rescore for rankings"""
        oSelector.fit(mXRemoved, vY)
        vScores = -numpy.log10(oSelector.pvalues_)
        """This indicates the relative importance of the metrics in
        where the noisier terms occur first"""        
        self.vFScoreRanks = vScores.argsort()
                
        """SVC fit"""
        """Training subset"""
        mXTrain = mXRemoved[0:800,:]
        vYTrain = vY[0:800]
        """The default C=1 here yields 99% and 96% success for training and cross-validation prediction"""
        oSVCModel = svm.SVC()
        oSVCModel.fit(mXTrain, vYTrain) 

        vbPredTrain = oSVCModel.predict(mXTrain)
        trainSuccess = 1 - abs(vbPredTrain - vYTrain).mean()

        """Cross-validation subset"""
        mXCross = mXRemoved[800:1000,:]
        vYCross = vY[800:1000]

        vbPredCross = oSVCModel.predict(mXCross)
        crossSuccess = 1 - abs(vbPredCross - vYCross).mean()
       
        vSuccess = [trainSuccess, crossSuccess]

        """Dimensionality reduction with PCA"""
        mCovariance = numpy.corrcoef(mXTrain.T)
        mU, vS, mV = numpy.linalg.svd(mCovariance, full_matrices=True)
        vS = vS/vS.sum(axis=0)

        """ The dimension may be reduced to k = 34 features by 
        orthogonal projection."""
        for k in range(1,len(vS)):  #to iterate between 10 to 20
            fove = vS[0:k].sum(axis=0)
            if fove > 0.99:
                print 'FOVE %f with %d features' % (fove,k)
                break #done

        """ However, there is no need to reduce the dimension further
        because the problem is well behaved within memory contraints
        with the 45 features after redundancy removal. """
        
        self.vSuccess = vSuccess