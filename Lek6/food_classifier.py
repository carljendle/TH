import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from random import shuffle
from plot_exp_var import *

np.random.seed(39)

#Läs in och inspektera datan
df = pd.read_csv("nndb_flat.csv")
print(df.head())


#Skräp - släpp det vi vet att vi inte använder för våra predictions och plocka bort NA-värden
cols_to_drop = ["CommonName", "MfgName","ScientificName", "ID", "ShortDescrip", "Descrip"]
df = df.drop(cols_to_drop, axis = 1)

df = df.dropna()

#Ger oss en överblick av klassbalans - printar sedan value counts för varje observerad klass
sns.displot(df, x = "FoodGroup")
plt.show()
print(df['FoodGroup'].value_counts())

#Vi vill alltså göra predictions på matgrupper - tar därför och encodar och poppar dessa.
food_encoder = LabelEncoder()
df["FoodGroup"] = food_encoder.fit_transform(df['FoodGroup'])
labels = df.pop("FoodGroup")

#Kollar korrelationer - ser till en början nästan ut att vara felaktigt (väldigt starka korrelationer mellan vissa variabler), men
#skrapar vi på ytan ser vi att de tekniskt sett beskriver samma sak (t.ex. Vitamin C och rekommenderat dagligt intag av Vitamin C.)
#Random forests som används är förvisso ganska robust mot oviktiga variabler, men hög korrelation kan vara god grund 
#för dimensionsreducering ändå.
sns.heatmap(df.corr())
plt.show()
labels = np.asarray(labels)
names = df.columns


data = np.asarray(df)
scaler = StandardScaler()

data = scaler.fit_transform(data)

#Vi ser även att vi bör kunna uttrycka all information med färre parametrar
plot_explained_variance(data, plot_range = 38, sum_range = 37)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

print(X_train.shape)

clf = RandomForestClassifier(max_depth=100, random_state=0)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

plot_importance(clf, names)

#Vi testar att droppa de variabler som tekniskt sett bara är dubblerade

cols_to_drop = list(filter(lambda x: x[-5:] == "USRDA", list(df.columns)))
df = df.drop(cols_to_drop, axis = 1)

data = np.asarray(df)
data = scaler.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
clf = RandomForestClassifier(max_depth=100, random_state=0)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
names = list(df.columns)
plot_importance(clf, names)


