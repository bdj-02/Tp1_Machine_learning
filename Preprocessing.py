## Importation des librairies
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


## Importation des données
positif_value = pd.read_pickle(r'data/imdb_raw_pos.pickle')
negatif_value = pd.read_pickle(r'data/imdb_raw_neg.pickle')

pos_value = {'Critiques': positif_value}
neg_value = {'Critiques': negatif_value}

## Création de la DataFrame avec PANDAS
# Récupération des données via une DataFrame pandas

dataframe1 = pd.DataFrame(pos_value)
dataframe1['Sentiment'] = 'positif'
dataframe1['Reponse'] = 1
dataframe2 = pd.DataFrame(neg_value)
dataframe2['Sentiment'] = 'Negatif'
dataframe2['Reponse'] = 0

### Concaténation de la donnée sous forme de tableau
data = pd.concat([dataframe1, dataframe2])

## Mélange de données

# STUMBLE DATA
### On affecte un commentaire qui décrit un sentiment positif ou negatif pour chaque critiques 
### en fonction du sentiment on affecte la valeur 1 pour positif et 0 pour négatif

stumb = data.sample(frac=1).reset_index()
stumb = stumb.drop(['index'], axis=1)

X = stumb.Critiques
y = stumb.Reponse

## SPLIT DATA 

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size= 0.2)

## VECTORISATION
# Transfromation des données cathegorielles en quantitafives

vector = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 4)
vector.fit(X_train)

X_train_vector = vector.transform(X_train)
X_test_vector = vector.transform(X_test)

### SCALER DATA
# Mise à l'échelle ou noramalisation de données

from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler

#Accuracy using Logistic Regression Model
transformer = MaxAbsScaler().fit(X_train_vector)
transformer_Xtrain = transformer.transform(X_train_vector)

### FIRST MODEL Logistic Regression
    # Mise en place d'unn modèle de regression logistique 

#Accuracy using Logistic Regression Model
Log_Reg = LogisticRegression()
Log_Reg.fit(transformer_Xtrain, y_train)

### PREDICTION
y_prediction = Log_Reg.predict(X_test_vector)

### CONFUSION MATRIX

confus_matrix = confusion_matrix(y_test, y_prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=confus_matrix, display_labels=['Positif', 'Negatif'])
disp.plot()
plt.show()

# 2rd MODEL
# TEST MODEL LG

test = []
test.append(input())
test_vect = vector.transform(test)
predLabel = Log_Reg.predict(test_vect)
tags = ['Negative','Positive']