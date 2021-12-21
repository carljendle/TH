import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression



def get_precisions_recalls(actual, preds, title):
    plt.figure(figsize=(16,4))
    
    plt.subplot(1,2,1)

    precision_0 = np.sum((actual == 0) & (preds == 0)) / np.sum(preds == 0)
    precision_1 = np.sum((actual == 1) & (preds == 1)) / np.sum(preds == 1) if np.sum(preds == 1) != 0 else 0

    
    plt.bar([0,1], [precision_0, precision_1])
    plt.xticks([0,1], ['Class 0', 'Class 1'], fontsize=20)
    plt.yticks(np.arange(0,1.1,0.1), fontsize=14)
    plt.ylabel('Precision', fontsize=20)
    plt.title(f'Precision Class 0: {round(precision_0,2)}\nPrecision Class 1: {round(precision_1,2)}', fontsize=20)
    
    plt.subplot(1,2,2)
    recall_0 = np.sum((actual == 0) & (preds == 0)) / np.sum(actual == 0)
    recall_1 = np.sum((actual == 1) & (preds == 1)) / np.sum(actual == 1)
    
    plt.bar([0,1], [recall_0, recall_1])
    plt.xticks([0,1], ['Class 0', 'Class 1'], fontsize=20)
    plt.yticks(np.arange(0,1.1,0.1), fontsize=14)
    plt.ylabel('Recall', fontsize=20)
    plt.title(f'Recall Class 0: {round(recall_0,2)}\nRecall Class 1: {round(recall_1,2)}', fontsize=20)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.show()





def generate_data(n, k, weights, cutoff):
    X = np.random.random((n, k))
    labels = (X@weights < cutoff*np.mean(X@weights)).astype(int)
    return X,labels.flatten()



n_train = 10000
n_test = 10000
k = 10
cutoff = 0.5
weights = np.random.random((k,1))



X_train, labels_train = generate_data(n_train, k, weights, cutoff)
X_test, labels_test = generate_data(n_test, k, weights, cutoff)


print('Fraction of positive labels:', str(round(100*np.mean(labels_train),3)) + '%')

dummy_clf = DummyClassifier(strategy="most_frequent")

dummy_clf.fit(X_train, labels_train)

preds = dummy_clf.predict(X_test)

print(f"Accuracy score: {accuracy_score(labels_test, preds)}")
print(f"F1-score: {f1_score(labels_test,preds)}")

title = "No balancing, dummy classifier"
get_precisions_recalls(labels_test, preds,title)





clf = DecisionTreeClassifier()
clf.fit(X_train, labels_train)
preds = clf.predict(X_test)

title = "No balancing, Decision Tree:"
get_precisions_recalls(labels_test, preds,title)


#Kan tackla detta på två sätt - bland annat genom viktning:

weight_minority_class = np.sum(labels_train == 0) / np.sum(labels_train == 1)


print(weight_minority_class)


clf = DecisionTreeClassifier(class_weight={0:1, 1:weight_minority_class})
clf.fit(X_train, labels_train)
preds = clf.predict(X_test)

title = "Minority class weighted, Decision Tree:"

get_precisions_recalls(labels_test, preds, title)



#Går också att oversampla - vi concatenatar minoritetsklassern

indices_0 = np.where(labels_train == 0)[0]
indices_1 = np.where(labels_train == 1)[0]
indices = np.concatenate([indices_0, indices_1])


weights = np.empty(indices_0.shape[0] + indices_1.shape[0])
weights[:indices_0.shape[0]] = 1
weights[indices_0.shape[0]:] = weight_minority_class
weights = weights/np.sum(weights)


sampled_indices = np.random.choice(indices, indices.shape[0], p=weights)



X_train_oversampled = X_train[sampled_indices]
labels_train_oversampled = labels_train[sampled_indices]

print('Fraction of positive labels in oversampled data:', str(round(100*np.mean(labels_train_oversampled),3)) + '%')



clf = DecisionTreeClassifier()
#clf = LogisticRegression()
clf.fit(X_train_oversampled, labels_train_oversampled)

preds = clf.predict_proba(X_test)
adder = np.zeros((2,2))


preds = clf.predict(X_test)

title = "Minority class oversampled, Decision Tree:"
get_precisions_recalls(labels_test, preds, title)
