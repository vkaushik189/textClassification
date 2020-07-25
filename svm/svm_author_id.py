#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import svm
#clf = svm.SVC(kernel='linear')
clf = svm.SVC(kernel='rbf')

# these 2 lines make the training set smaller for faster training
#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]

t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

t1 = time()
pred = clf.predict(features_test)
print("testing time:", round(time()-t1, 3), "s")

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print("Base Accuracy: " + str(accuracy))

#trying differnt values of C
c_values = [10, 100, 1000, 10000]
for i in c_values:
    clf_new = svm.SVC(kernel='rbf', C = i)
    clf_new.fit(features_train, labels_train)
    pred_new = clf_new.predict(features_test)
    accuracy_new = accuracy_score(labels_test, pred_new)
    print("Accuracy when C = " + str(i) + " is: " + str(accuracy_new))


#########################################################


