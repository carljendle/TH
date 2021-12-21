from math import gamma
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



df = pd.read_csv("bikebuyers.csv")


#EDA

names = [col for col in df.columns]


print(df.head())


one_hot_columns = [name for name, dtype in zip(names,df.dtypes) if str(dtype) == 'object']
one_hot_data = pd.get_dummies(df[one_hot_columns],drop_first=True)

new_df = pd.read_csv("bikebuyers.csv")
for name in one_hot_columns:
    new_df.pop(name)

new_names = [name for name in one_hot_data.columns]
for name in new_names:
    new_df[name] = one_hot_data[name].values






labels = new_df.pop('BikeBuyer')



print(new_df.head())

names = [col for col in new_df.columns]

data = np.asarray(new_df)

X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.3)

best_score_all_features = -1
best_accuracy_all_features = -1
min_depth = 4
max_depth = 20
max_features = int(np.sqrt(len(new_df.columns)))
best_clf = None
for i in range(min_depth,max_depth):
    clf = RandomForestClassifier(n_estimators = 600, max_depth=i, random_state=0, max_features = max_features)
    clf.fit(X_train, Y_train)

    predictions = clf.predict(X_test)

    print(f"Results for forest:")
    print(f"Accuracy score: {accuracy_score(Y_test, predictions)}")
    print(f"F1-score: {f1_score(Y_test, predictions)}")

    if f1_score(Y_test, predictions) > best_score_all_features:
        best_score_all_features = f1_score(Y_test, predictions)
        best_clf = clf
    if accuracy_score(Y_test, predictions) > best_accuracy_all_features:
        best_accuracy_all_features = accuracy_score(Y_test, predictions)
        

print("--------------Results--------------")

print(f"All features best accuracy score: {best_accuracy_all_features}")
print(f"All features best f1_score: {best_score_all_features}")



importances = best_clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in best_clf.estimators_], axis=0)
forest_importances = pd.Series(importances, index=names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

plt.show()




result = permutation_importance(
    best_clf, X_test, Y_test, n_repeats=10, random_state=42, n_jobs=2
)



forest_importances = pd.Series(result.importances_mean, index=names)




fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()