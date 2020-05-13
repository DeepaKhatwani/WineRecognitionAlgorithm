# Python version
import sys
# scipy
import scipy
# numpy
import numpy as np
# matplotlib
import matplotlib

# pandas
import pandas as pd
# scikit-learn
from sklearn import datasets

#seaborn
import seaborn as sns
import matplotlib.pyplot as plt


from pandas.plotting import scatter_matrix


#Method 1 to know the target
#k, y = sklearn.datasets.load_wine(return_X_y=True)
# print(y)

#Method 2
############################################ To load data
raw_data = datasets.load_wine()
print(type(raw_data))
print(raw_data.feature_names)
print(raw_data.target_names)
df_feature = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
df = df_feature
df['target'] = raw_data['target']

df['class_target']=df['target'].map(lambda ind: raw_data['target_names'][ind])
df.head()


############################################ To explore data using pandas
df.shape # 178 rows, 15 columns
df.info() #No missing data found
df.columns

df.describe()
df.class_target.describe()
df.groupby('class_target').size()
df.groupby('class_target').describe().T

# Divide class wise data
data_class_0 = df[df.target==0]
print(data_class_0)
data_class_1 = df[df.target==1]
print(data_class_1)
data_class_2 = df[df.target==2]
print(data_class_2)

############################################ To explore data using plot

#------------------------------BOX PLOT ------------------------


#exec(df.alcohol.plot(kind='box'))
#plt.show()

#data = [data_class_0.alcohol, data_class_1.alcohol, data_class_2.alcohol]
#fig, ax = plt.subplots()
#ax.boxplot(data)

#df.boxplot(column = 'alcohol', by = 'target');
#plt.title('')

#Method 1

for col_name in df.columns:
    print(col_name)
    statement = "df.boxplot(column = '" + col_name + "', by = 'target')"
    exec(statement)    
    plt.show()

#Method 2
    '''
for col_name in df.columns:
    print(col_name)
    statement1 = "data = [data_class_0." + col_name + ", data_class_1." + col_name + ", data_class_2." + col_name + "]"
    exec(statement1)
    fig, ax = plt.subplots()
    ax.set_title(col_name)
    ax.boxplot(data)
 '''
#Method 3
'''    
for col_name in df.columns:
    print(col_name)
    statement = "df." + col_name + ".plot(kind='box')"
    exec(statement)    
    plt.show()
'''


#------------------------------HISTOGRAM PLOT ------------------------


#df.hist(column = 'alcohol', by = 'target');
#plt.show()
for col_name in df.columns:
    print(col_name)
    statement = "df.hist(column = '" + col_name +"', by = 'target');"
    exec(statement)    
    plt.show()


#------------------------------ SCATTER PLOT ------------------------
scatter_matrix(df,figsize=(10,8))
plt.show()

#------------------------------ CHECKING RELATIONS ------------------------
df.groupby('class_target').get_group('class_0').boxplot(figsize=(8,6))
plt.show()

df.groupby('class_target').get_group('class_1').boxplot(figsize=(8,6))
plt.show()

df.groupby('class_target').get_group('class_2').boxplot(figsize=(8,6))
plt.show()

colors = {'class_0':'b', 'class_1':'r','class_2':'y'}

plt.figure(figsize=(12,10))
sns.pairplot(df, hue='class_target', palette = colors)

plt.figure(figsize=(12,10))
sns.scatterplot('alcohol','hue', hue='class_target', data=df, palette = colors)

#------------------------------ CHECKING RELATIONS ------------------------

corr_matrix = df.corr()
print(corr_matrix["target"].sort_values(ascending=False))


#colum_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
# Correlation matrix
correlations = df.corr()
# Plot figsize
fig, ax = plt.subplots(figsize=(10, 10))
# Generate Color Map
colormap = sns.diverging_palette(220, 10, as_cmap=True)
# Generate Heat Map, allow annotations and place floats in map
sns.heatmap(correlations, cmap=colormap, annot=True, fmt=".2f")
ax.set_xticklabels(
    df.columns,
    rotation=45,
    horizontalalignment='right'
);
ax.set_yticklabels(df.columns);
plt.show()



#Observations
# Alcohot - class_1 has lower amount of alcohol
# Malic_acid - class_2 has more malic_acid
# ash - class_1 has comparitively lower ash
# total_phenols, flavonids - perfect to distinguish class


############################################ Split Data into train and test data
#The industry standard

data = df.values
type(data) #always use this format

X = data[:,:-2]
X

y = data[:,-2]
y=y.astype('int')
y

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)
X_train.shape
X_test.shape
y_train.shape
y_test.shape



############################################ Applying MACHINE LEARNING ALGORITHM TO TRAIN DATA

#------------------- Machine learning 1 ------------
print('---------- LogisticRegression--------------------------')
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine species for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#------------------- Machine learning 2 ------------
print('---------- KNeighborsClassifier--------------------------')
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 3 ------------
print('---------- DecisionTreeClassifier--------------------------')

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 4 ------------
print('---------- SVC--------------------------')

from sklearn.svm import SVC

classifier = SVC()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 5 ------------
print('---------- GaussianNB--------------------------')

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 6 ------------
print('---------- RandomForestClassifier--------------------------')

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 7 ------------
print('---------- MLPClassifier--------------------------')

from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


################################################################################
# Answer : DecisionTreeClassifier is best machine learning algorithm for this example





