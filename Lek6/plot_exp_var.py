import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from typing import List
def plot_explained_variance(train_data: pd.DataFrame, plot_range: int = 300, sum_range: int = 10) -> None:
    '''
    Plots the explained_variance for the range of 

    Args in: train_data - data to fit PCA
             plot_range - number of principal components to include in the sum of explained variances
             sum_range - number of principal compontens explained variances to sum and print
    Returns: None
    '''
    pca = PCA(plot_range)
    pca_full = pca.fit(train_data)

    print(f'Sum of the {sum_range} most important features:{sum(pca_full.explained_variance_ratio_[:sum_range])}')

    plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
    plt.xlabel('# of components')
    plt.ylabel('Cumulative explained variance')
    plt.title("Amount of total variance included in the principal components")
    plt.show()

def plot_importance(classifier: RandomForestClassifier, names: List[str]):
    importances = classifier.feature_importances_

    forest_importances = pd.Series(importances, index=names)
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()