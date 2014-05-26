""" Groups and Topics

The objective of this task is to explore the structure of the deals.txt file. 

Building on task 1, we now want to start to understand the relationships to help us understand:

1. What groups exist within the deals?
2. What topics exist within the deals?

"""

from gensim import corpora, models, similarities
#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
#                    level=logging.INFO)

class Task2:

    """ Return the objectives of the task as member 
        functions.

        """

    def __init__(self):
        pFile = '../data/deals.txt'

        oDict = corpora.Dictionary(line.lower().split() for line in open(pFile))

        msDocs = []
        """Read bad deals in"""
        for line in open(pFile, 'r'):
            msDocs.append(line.lower().split())

        oDict = corpora.Dictionary(msDocs)
        oCorpus = [oDict.doc2bow(text) for text in msDocs]

        """TF-IDF transformation provides better weightings using word-frequencies."""
        oTFIDF = models.TfidfModel(oCorpus)
        oCorpusTFIDF = oTFIDF[oCorpus]

        """Initialize an LSI transformation
            
        I choose 20 topics for computational convenience. I could use the elbow method
        and cluster dispersion of k-means to identify number of clusters, but that isn't
        reliable. There's also the formula 1/(fraction of nonzero entries), but that
        yields a very large number in this case."""
        self.oLSIModel = models.LsiModel(oCorpusTFIDF, id2word=oDict, num_topics=20)
        """Print top five topics"""
        oLSI.print_topics()        