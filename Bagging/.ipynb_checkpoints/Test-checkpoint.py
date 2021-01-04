import numpy as np
import mltools as ml
from Bagging import Bagging
from Guassian_Bayes_Pvalue import TheoreticalGaussianClassifier
from PCA_Pvalue_Classifier import PCA_pvalue
from Distance_Classifier import Distance_classifier

import time

iris = np.genfromtxt("iris.txt", delimiter=None)
Y = np.array([int(i) for i in iris[:, -1]])
X = iris[:, 0:-1]
Xtr, Xva, Ytr, Yva = ml.splitData(X, Y, 0.8)


print(Xtr.shape)

# t0 = time.time()
# model = Bagging()
# model.train(Xtr, Ytr, percentageForClassifiers=[0.8, 0.1, 0.1], nBag=10, n_boot=1000)
# print("Time:", time.time() - t0)
# print(model.predict_all_pvalue(Xva))

# model = TheoreticalGaussianClassifier(Xtr, Ytr)
# print(model.all_p_values(Xva))

# model = PCA_pvalue(Xtr, Ytr)
# print(model.all_p_values(Xva))

model = Distance_classifier(Xtr, Ytr, model = "gamma", threshold = 1/len(X))
model.fit()
print(model.all_p_values(Xva))