import sys
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from csv import reader

def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    dataset = np.array(dataset).astype('float')
    predictors = dataset[:,0:13]
    labels = dataset[:,13].astype('int')
    return predictors, labels

filename = sys.argv[1]
X_data, Y_data = load_csv(filename)
