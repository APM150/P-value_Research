from Distance_Classifier import Distance_classifier
from Guassian_Bayes_Pvalue import EmpiricalGaussianClassifier
from Guassian_Bayes_Pvalue import TheoreticalGaussianClassifier
from PCA_Pvalue_Classifier import PCA_pvalue
import mltools as ml
import numpy as np
import random


class Bagging(ml.base.classifier):
    def __init__(self):
        self.bags = None
        self.classes = None


    def train(self, X, Y, percentageForClassifiers: list, nBag, n_boot):
        """
        :param X:
        :param Y:
        :param percentageForClasses: percentage for each of the classifier [distance, gaussian, PCA]
        :param nBag:
        :param n_boot:
        :return:
        """
        self.bags = [None] * nBag
        self.classes = np.unique(Y)
        for i in range(nBag):
            Xb, Yb = ml.bootstrapData(X, Y, n_boot=n_boot)
            probability = random.random()
            if probability <= percentageForClassifiers[0]: # Distance classifier
                self.bags[i] = Distance_classifier(Xb, Yb, model="gamma", threshold=1 / len(X))
                self.bags[i].fit()
            elif percentageForClassifiers[0] < probability <= percentageForClassifiers[0] + percentageForClassifiers[1]: # Gaussian classifier
                self.bags[i] = EmpiricalGaussianClassifier(Xb, Yb)
            elif percentageForClassifiers[0] + percentageForClassifiers[1] < probability < sum(percentageForClassifiers): # PCA classifier
                self.bags[i] = PCA_pvalue(Xb, Yb)


    def all_p_values(self, X):
        """
        :param X: m x number of features
        :return: m x number of classes p values
        """
        pValues = np.zeros((X.shape[0], len(self.classes)))
        for i in range(len(self.bags)):
            print("Bag", i+1)
            pValues += self.bags[i].all_p_values(X)
        return pValues / len(self.bags)


    def score(self, X, Y, all_p_vals="None"):
        """
        X: r by c
        threshold: pvalue threshold
        return: the accuracy
        """
        if all_p_vals == "None":
            all_p_vals = self.all_p_values(X)
        return np.sum(np.argmax(all_p_vals, axis=1) == Y) / X.shape[0]

