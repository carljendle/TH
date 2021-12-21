import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


df = pd.read_csv("bikebuyers.csv")

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


#Kan allt göras kompaktare!
state_provinces = list(filter(lambda x: x[0:5] == "State", list(new_df.columns)))

state_provinces = list(filter(lambda x: x[-2:] != "CA", state_provinces))
state_provinces = list(filter(lambda x: x[-2:] != "WA", state_provinces))

#Poppa lite oviktiga parametrar
for state_province in state_provinces:
   new_df.pop(state_province)
new_df.pop("PostalCode")

labels = new_df.pop('BikeBuyer')



print(new_df.head())

names = [col for col in new_df.columns]

data = np.asarray(new_df)

X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.3, random_state = 42)

best_score_all_features = -1
best_accuracy_all_features = -1
best_clf = None
best_settings = {"max_depth": 0, "n_estimators":0}
min_depth = 6
max_depth = 15
n_estimators = [400+100*x for x in range(5)]
max_features = int(np.sqrt(len(new_df.columns)))
for i in range(min_depth, max_depth):
    for estimators in n_estimators:
        clf = RandomForestClassifier(n_estimators = estimators, max_depth=i, random_state=0, max_features = int(np.sqrt(len(list(new_df.columns)))))
        clf.fit(X_train, Y_train)

        predictions = clf.predict(X_test)

        print(f"Results for forest:")
        print(f"Accuracy score: {accuracy_score(Y_test, predictions)}")
        print(f"F1-score: {f1_score(Y_test, predictions)}")
        #

        if f1_score(Y_test, predictions) > best_score_all_features:
            best_score_all_features = f1_score(Y_test, predictions)
            best_accuracy_all_features = accuracy_score(Y_test, predictions)
            best_clf = clf
            best_settings["max_depth"] = i
            best_settings["n_estimators"] = estimators

            

print("--------------Results--------------")

print(f"All features best accuracy score: {best_accuracy_all_features}")
print(f"All features best f1_score: {best_score_all_features}")
print(f"Best settings: {best_settings}")


importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
forest_importances = pd.Series(importances, index=names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

plt.show()