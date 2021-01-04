from scipy.stats import multivariate_normal
import mltools as ml
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import euclidean_distances


class TheoreticalGaussianClassifier:

    def __init__(self, X="None", Y="None"):
        if X == "None" or Y == "None":
            return
        self.fit(X, Y)

    def fit(self, X, Y):
        self.classes, self.counts = np.unique(Y, return_counts=True)  # [A, B, ...],  [numA, numB, ..]
        self.centers = np.array([np.mean(X[Y == c, :], axis=0) for c in self.classes])  # [centerA, centerB, ...]
        self.covs = np.array([np.cov(X[Y == c, :].T) for c in self.classes])  # [covA, covB, ...]
        self.distributions = np.array([multivariate_normal(mean=self.centers[i], cov=self.covs[i], allow_singular=True) for i, c in
                                       enumerate(self.classes)])  # [disA, disB, ...]

    def p_value(self, x):
        """
        x: feature vector
        return: p value for each of the class
        """
        return np.array(
            [(self.distributions[i].cdf(x) if self.distributions[i].cdf(x) < 0.5 else 1 - self.distributions[i].cdf(x))
             for i in range(len(self.classes))])


    def all_p_values(self, X):
        """
        X: r by c
        return: p values of each of the class for all data points
        """
        return np.array([self.p_value(x) * 2 for x in X])


    def predictClass(self, X):
        """
        X: r by c
        return: predcit the class that has the highest p-value
        """
        return np.argmax(self.all_p_values(X), axis=1)

    def score(self, X, Y, threshold=0):
        """
        X: r by c
        threshold: pvalue threshold
        return: the accuracy
        """
        all_p_vals = self.all_p_values(X)
        return np.sum(np.argmax(all_p_vals, axis=1) == Y) / X.shape[0]



class EmpiricalGaussianClassifier:
    
    def __init__(self, X="None", Y="None"):
        if X == "None" or Y == "None":
            return
        self.fit(X, Y)
        
        
    def fit(self, X, Y):
        self.classes, self.counts = np.unique(Y, return_counts=True)  #[A, B, ...],  [numA, numB, ..]
        self.model = GaussianNB()
        self.model.fit(X, Y)
#         self.centers = np.array([np.mean(X[Y==c, :], axis=0) for c in self.classes]) # [centerA, centerB, ...]
#         self.distances = np.array([euclidean_distances(X[Y==c], [centers[i,:]]).T[0] for (i, c) in zip(range(len(self.centers)), self.classes)])  # [distancesA, distancesB, ...]
        allProbas = self.model.predict_proba(X)
        self.probas = np.array([allProbas[Y==c, i] for i, c in enumerate(self.classes)])  # [probasForClassA, probasForClassB, ..]
        
        
           
    def p_value(self, x):
        """
        x: feature vector
        return: p value for each of the class
        """
        xProba = self.model.predict_proba(np.atleast_2d(x))[0]
        p_val = np.array([(np.sum(xProba[i] >= self.probas[i]) / self.counts[i] if np.sum(xProba[i] >= self.probas[i]) / self.counts[i] < 0.5 else 1 - np.sum(xProba[i] >= self.probas[i]) / self.counts[i]) for i in range(len(self.classes))])
        return p_val
    
    
    def all_p_values(self, X):
        """
        X: r by c
        return: p values of each of the class for all data points
        """
        return np.array([self.p_value(x) for x in X])
    
    
    def predictProba(self, x):
        """
        x: feature vector
        return: probability for each of the class
        """
        return self.model.predict_proba(np.atleast_2d(x))[0]
    
    
    def all_probas(self, X):
        """
        X: r by c
        return: all the proabability for each of the class for all data points
        """
        return np.array([self.predictProba(x) for x in X])
    
    
    def predictClass(self, X):
        """
        X: r by c
        return: predcit the class that has the highest p-value
        """
        return np.argmax(self.all_p_values(X), axis=1)
        
        
    
    def score(self, X, Y, all_p_vals="None"):
        """
        X: r by c
        threshold: pvalue threshold
        return: the accuracy
        """
        if all_p_vals == "None":
            all_p_vals = self.all_p_values(X)
        return np.sum(np.argmax(all_p_vals, axis=1) == Y) / X.shape[0]
