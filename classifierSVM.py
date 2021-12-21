from math import gamma
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline



df = pd.read_csv("bikebuyers.csv")


#EDA

names = [col for col in df.columns]


print(df.head())
df.pop("PostalCode")
df.pop('StateProvinceCode')

one_hot_columns = [name for name, dtype in zip(names,df.dtypes) if str(dtype) == 'object']
one_hot_data = pd.get_dummies(df[one_hot_columns],drop_first=True)

new_df = pd.read_csv("bikebuyers.csv")
for name in one_hot_columns:
    new_df.pop(name)

new_names = [name for name in one_hot_data.columns]
for name in new_names:
    new_df[name] = one_hot_data[name].values

new_df.pop("StateProvinceCode")
#new_df.pop("PostalCode")




labels = new_df.pop('BikeBuyer')

print(new_df.head())
names = [col for col in new_df.columns]

data = np.asarray(new_df)
#Grid search
kernels = ['rbf']
gammas = [1.2, 0.5, 0.3]
cs = [0.2*x + 1 for x in range(30)]
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

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

            print(f"Results for kernel {kern}, C parameter {c}, gamma {gamma}")
            print(f"Accuracy score: {accuracy_score(Y_test, predictions)}")
            print(f"F1-score: {f1_score(Y_test, predictions)}")
            #

            if f1_score(Y_test, predictions) > best_score_all_features:
                best_score_all_features = f1_score(Y_test, predictions)
                settings_best_f1_score = kern + ", " + str(c) + ", " + str(gamma)
            if accuracy_score(Y_test, predictions) > best_accuracy_all_features:
                best_accuracy_all_features = accuracy_score(Y_test, predictions)
                settings_best_accuracy = kern + ", " + str(c) + ", " + str(gamma)
        

print("--------------Results--------------")

print(f"All features best f1_score: {best_score_all_features}")
print(f"All features best accuracy score: {best_accuracy_all_features}")
print(settings_best_accuracy)
print(settings_best_f1_score)






