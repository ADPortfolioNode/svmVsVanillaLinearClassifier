import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()

target = digits.target
flatten_digits = digits.images.reshape((len(digits.images), -1))

#   

#visualize some hand written images in the dataset

_, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 4))
for ax, image, label in zip(axes, digits.images, target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('%i' % label)
print("Some hand-written images in the dataset..")
#split data into training and testing sets
print("Splitting data into training and testing sets..")
X_train, X_test, y_train, y_test = train_test_split(flatten_digits, target, test_size=0.5, shuffle=False)
print("Data split successfully..", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#Hand-written classification with Logistic Regression
scaler = StandardScaler()
X_train_logistic = scaler.fit_transform(X_train)
X_test_logistic = scaler.transform(X_test)
print("Creating and fitting the model..", X_train_logistic.shape, y_train.shape)

logit = LogisticRegression(C=0.01, penalty='l1', solver='saga', tol=0.1, multi_class='multinomial')
logit.fit(X_train_logistic, y_train)
print("Model created and fitted successfully..", logit)
#predict the labels of the test set

y_pred_logistic = logit.predict(X_test_logistic)
print("Predictions made successfully..", y_pred_logistic)
#get accuracy of the model
print("Accuracy: "+str(logit.score(X_test_logistic, y_test)))

#confusion matrix
label_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
cmx = confusion_matrix(y_test, y_pred_logistic, labels=label_names)
print("Confusion matrix: ", cmx)

df_cm = pd.DataFrame(cmx)
# plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
title = "Confusion Matrix for SVM results"
plt.title(title)
plt.show()
print("Confusion matrix displayed successfully..")
#Hand-written classification with SVM
#create and fit the model

svm_classifier = svm.SVC(gamma='scale')
print("Creating and fitting the model..", X_train.shape, y_train.shape)
svm_classifier.fit(X_train, y_train)
print("Model created and fitted successfully..", svm_classifier)
#predict the labels of the test set

y_pred_svm = svm_classifier.predict(X_test)
print("Predictions made successfully..", y_pred_svm)
#get accuracy of the model
print("Accuracy: "+str(accuracy_score(y_test, y_pred_svm)))
print("Accuracy:(svm classifier score->) "+str(svm_classifier.score(X_test, y_test)))
 
#let's look at the matrix
label_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
cmx = confusion_matrix(y_test, y_pred_svm, labels=label_names)
df_cm = pd.DataFrame(cmx)
print("Confusion matrix: ", cmx)
# plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
title = "Confusion Matrix for SVM results"
plt.title(title)
plt.show()

#compore the two models with k-fold cross validation
print("Comparing the two models with k-fold cross validation..")
#k-fold Cross validation is used when there are limited samples, the handwritten dataset contains about 1800 samples, this will give an opportunity for all the data to be in the training and test set at different given times. We will add l2 regularization to visualize how well they both do against SVM.

algorithm = []
algorithm.append(('SVM', svm_classifier))
algorithm.append(('Logistic_L1', logit))
algorithm.append(('Logistic_L2', LogisticRegression(C=0.01, penalty='l2', solver='saga', tol=0.1, multi_class='multinomial')))
print("Algorithm created successfully..")

results = []
names = []
y = digits.target
print("Performing k-fold cross validation..")
for name, algo in algorithm:
# Option 1: Set shuffle=True
    # train_test_split(your_data, shuffle=True, random_state=your_random_state)

# Option 2: Leave random_state=None
    # train_test_split(your_data, shuffle=False)
    k_fold = model_selection.KFold(n_splits=10, shuffle=True, random_state=0)
    if name == 'SVM':
        X = flatten_digits
        cv_results = model_selection.cross_val_score(algo, X, y, cv=k_fold, scoring='accuracy')
        print("SVM: ", cv_results)
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(flatten_digits)
        cv_results = model_selection.cross_val_score(algo, X, y, cv=k_fold, scoring='accuracy')
        print("Logistic: ", cv_results)
        
    results.append(cv_results)
    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())   
    print(msg)
print("k-fold cross validation done successfully..")
#We plot and we can see that SVM performs better all the time even with k-fold cross validation and it is better than both Logistic regressions on average
fig = plt.figure()
fig.suptitle('Compare Logistic and SVM results')
ax = fig.add_subplot()
plt.boxplot(results)
plt.ylabel('Accuracy')
ax.set_xticklabels(names)
plt.show()    

# Path: svmVsVanillaLInearClassifier.py
print("~~~~~~~End of Line~~~~~~~")