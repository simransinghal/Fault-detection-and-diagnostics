import sys
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import numpy as np
from csv import reader

def load_csv(filename):
    predictors = np.genfromtxt(filename, delimiter = ',')[:,:-1]
    #print predictors[0][1]
    labels = np.array(list(reader(open(filename, "rt"))))[:,8]
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(predictors)
    predictors = imp.transform(predictors)
    predictors = preprocessing.scale(predictors)
    return predictors, labels

def SVM(train_data, train_labels, test_data, test_labels):
    print (train_data.shape), (test_data.shape)
    C_range = np.outer(np.logspace(-1, 1, 3),np.array([1,5]))
    print C_range
    C_range = C_range.flatten()

    gamma_range = np.outer(np.logspace(-3, 0, 4),np.array([1,5]))
    gamma_range = gamma_range.flatten()
    parameters = {'kernel':['sigmoid'], 'C':C_range, 'gamma': gamma_range}
    svm_clsf = svm.SVC()

    grid_clsf = GridSearchCV(estimator=svm_clsf,param_grid=parameters,n_jobs=1, verbose=2, cv=10)
    grid_clsf.fit(train_data, train_labels)

    classifier = grid_clsf.best_estimator_
    print classifier

    train_predictions = classifier.predict(train_data)

    train_cm = confusion_matrix(train_labels, train_predictions)
    print train_cm

    train_accuracy = accuracy_score(train_labels, train_predictions)

    print "Training Accuracy: %.4f" % (train_accuracy)

    test_predictions = classifier.predict(test_data)

    cm = confusion_matrix(test_labels, test_predictions)
    print cm

    accuracy = accuracy_score(test_labels, test_predictions)
    print "Test Accuracy: %.4f" % (accuracy)

    precision = precision_score(test_labels, test_predictions, average='weighted')
    recall = recall_score(test_labels, test_predictions, average='weighted')
    f1 = 2.0 * (precision * recall) / (precision + recall)

    print "Test Precision: %.4f" % (precision)
    print "Test Recall: %.4f" % (recall)
    print "Test f1_score: %.4f" % (f1)

    return accuracy, precision, recall, f1

filename = sys.argv[1]
X_data, Y_data = load_csv(filename)
#print X_data
#print X_data.shape

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30)
metrics = []
fold = 1
for train_indices, test_indices in sss.split(X_data, Y_data):
    train_data, test_data = X_data[train_indices], X_data[test_indices]
    train_labels, test_labels = Y_data[train_indices], Y_data[test_indices]
    metrics.append(SVM(train_data, train_labels, test_data, test_labels))
    fold += 1

accuracy = 0.00
precision = 0.00
recall = 0.00
fi = 0.00
for i in metrics:
        accuracy += i[0]
        precision += i[1]
        recall += i[2]
        fi += i[3]
accuracy = accuracy/1.0
precision = precision/1.0
recall = recall/1.0
fi = fi/1.0
print (accuracy),(","),(precision),(", "),(recall),(", "),(fi)

