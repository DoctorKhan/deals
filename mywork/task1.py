#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Features

The objective of this task is to explore the corpus, deals.txt. 

The deals.txt file is a collection of deal descriptions, separated by a new line, from which 
we want to glean the following insights:

1. What is the most popular term across all the deals?
2. What is the least popular term across all the deals?
3. How many types of guitars are mentioned across all the deals?

"""

from gensim import corpora, models, similarities
import re


class Task1:

    """ Return the three objectives of the task as member 
        functions.

        """

    def __init__(self, pDocs=None):
        """Location of deals data"""
        if pDocs is None:
            self.pDocs = '../data/deals.txt'
        else:
            self.pDocs = pDocs

    def most(self):
        """Identify the most common term"""
        oMost = corpora.Dictionary(line.lower().split() for line in
			open(self.pDocs))
        oMost.filter_extremes(no_below=1, no_above=1, keep_n=1)
        sMost = list(oMost.token2id)[0]
        return sMost

    def least(self):
        """Identify the least common term"""
        oLeast = corpora.Dictionary(line.lower().split() for line in
			open(self.pDocs))
        once_ids = [tokenid for (tokenid, docfreq) in
                    oLeast.dfs.iteritems() if docfreq == 1]
        oLeast.filter_tokens(good_ids=once_ids)
        sLeast = list(oLeast.token2id)[0]
        return sLeast

    def guitars(self):
        """Return the number of types of guitars"""
        rGuitars = re.compile(r"\w+ [Gg]uitar")
        oGuitars = corpora.Dictionary(rGuitars.findall(line)
                for line in open(self.pDocs))
        nGuitars = len(oGuitars)
        return nGuitars


