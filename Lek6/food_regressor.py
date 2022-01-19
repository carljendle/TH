import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from plot_exp_var import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error

np.random.seed(42)
df = pd.read_csv("nndb_flat.csv")

print(df.head())

#Skräp - släpp det vi vet att vi inte använder för våra predictions och plocka bort NA-värden
cols_to_drop = ["CommonName", "MfgName","ScientificName", "ID", "ShortDescrip", "Descrip", "FoodGroup"]
df = df.drop(cols_to_drop, axis = 1)


print(df.shape)

df = df.dropna()

#Värden för regression
labels = df.pop("Energy_kcal")
sns.heatmap(df.corr())
plt.show()
labels = np.asarray(labels)

names = df.columns

df = np.asarray(df)
scaler = StandardScaler()
df = scaler.fit_transform(df)

X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.3)
reg = Lasso(alpha=1)
reg.fit(X_train, y_train)


print('R squared training set', round(reg.score(X_train, y_train)*100, 2))
print('R squared test set', round(reg.score(X_test, y_test)*100, 2))



# Training data
pred_train = reg.predict(X_train)
mse_train = mean_squared_error(y_train, pred_train)
print('MSE training set', round(mse_train, 2))

# Test data
pred = reg.predict(X_test)
mse_test =mean_squared_error(y_test, pred)
print('MSE test set', round(mse_test, 2))

alphas = np.linspace(0.01,500,100)
lasso = Lasso(max_iter=10000)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('Standardized Coefficients')
plt.title('Lasso coefficients as a function of alpha')
plt.show()


# Lasso with 5 fold cross-validation
model = LassoCV(cv=5, random_state=0, max_iter=10000)
plt.hist(y_train, bins = 100)
plt.title("Training data bin count")
plt.show()
model.fit(X_train, y_train)
lasso_best = Lasso(alpha=model.alpha_)
lasso_best.fit(X_train, y_train)



print('R squared training set', round(lasso_best.score(X_train, y_train)*100, 2))
print('R squared test set', round(lasso_best.score(X_test, y_test)*100, 2))


pred = lasso_best.predict(X_test)



mse_test =mean_squared_error(y_test, pred)

print(mse_test)

#Verkar i synnerhet ha svårt för lågkalorialternativ! Tydlig linjär trend då vi tekniskt sett vet att kalorier alltid kommer
# att vara en linjärkombination av våra features - även en god anledning till varför LASSO fungerar bra.
plt.scatter(pred, y_test)
plt.title("Scatter plot for predictions")
plt.show()


