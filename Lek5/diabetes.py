import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


#Skulle kunna testas - SMOTE eller annan typ utav data augmentation.
#Logistic Regression, förmodligen sämre performance
#Polynomial kernel - kanske!

data = pd.read_csv("diabetes.csv")


labels = data.pop('Outcome')

data = np.asarray(data)

X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.3, random_state = 0)

n_estimators = [50, 100, 150, 250, 300]
learning_rate = [0.15, 0.2, 0.25]
max_depth = [8, 9, 10,11]

best_f1 = -1
best_f1_settings = None
best_accuracy = -1
for est in n_estimators:
    for lr in learning_rate:
        for depth in max_depth:
            clf = GradientBoostingClassifier(n_estimators=est, learning_rate=lr, max_depth=depth, random_state=0).fit(X_train, Y_train)

            predictions = clf.predict(X_test)
            if f1_score(Y_test, predictions) > best_f1:
                best_f1 = f1_score(Y_test, predictions)
                best_accuracy = accuracy_score(Y_test, predictions)
                best_f1_settings = "Depth: " + str(depth) + " lr:" + str(lr) + "Estim: " + str(est)

print("--------------Results Gradient Boosting:--------------")
print(f"Settings best Gradient Booster: {best_f1_settings}")
print(f"Best F1-score from Gradient Boosting: {best_f1}")
print(f"Accuracy: {best_accuracy}")





best_score_all_features = -1
best_accuracy_all_features = -1
min_depth = 4
max_depth = 20
n_obs, n_features = X_train.shape
max_features = int(np.sqrt(n_features))
best_clf = None
best_settings = {"max_depth": 0, "n_estimators":0}
for i in range(min_depth, max_depth):
    for estimators in n_estimators:
        clf = RandomForestClassifier(n_estimators = estimators, max_depth=i, random_state=0, max_features = max_features)
        clf.fit(X_train, Y_train)

        predictions = clf.predict(X_test)

        #

        if f1_score(Y_test, predictions) > best_score_all_features:
            best_score_all_features = f1_score(Y_test, predictions)
            best_accuracy_all_features = accuracy_score(Y_test, predictions)
            best_clf = clf
            best_settings["max_depth"] = i
            best_settings["n_estimators"] = estimators
        

print("--------------Results Random Forest:--------------")

print(f"All features best accuracy score: {best_accuracy_all_features}")
print(f"All features best f1_score: {best_score_all_features}")





kernels = ['rbf']
gammas = [2,1, 0.5, 0.3, 0.2]
cs = [0.2*x + 1 for x in range(40)]

best_score_all_features = -1
best_accuracy_all_features = -1
settings_best_accuracy = None
settings_best_f1_score = None

for kern in kernels:
    for c in cs:
        for gamma in gammas:
            pipe = Pipeline([
            ('scale', StandardScaler()),
            ('clf', SVC(kernel = kern, C = c, gamma = gamma))])


            pipe.fit(X_train, Y_train)

            predictions = pipe.predict(X_test)

            #

            if f1_score(Y_test, predictions) > best_score_all_features:
                best_score_all_features = f1_score(Y_test, predictions)
                settings_best_f1_score = kern + ", " + str(c) + ", " + str(gamma)
            if accuracy_score(Y_test, predictions) > best_accuracy_all_features:
                best_accuracy_all_features = accuracy_score(Y_test, predictions)
                settings_best_accuracy = kern + ", " + str(c) + ", " + str(gamma)
        

print("--------------Results SVM--------------")

print(f"All features best f1_score: {best_score_all_features}")
print(f"All features best accuracy score: {best_accuracy_all_features}")
print(settings_best_accuracy)
print(settings_best_f1_score)