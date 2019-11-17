#!/usr/bin/env python
# coding: utf-8

# ## Charging required librarys

# In[ ]:


import pandas as pd
import numpy as np
import os
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.metrics as metrics
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing datasets, first look and concat

# In[ ]:


# Changing directory and importing databases
os.chdir('C:\\Users\\Wellington\\Jupyter notebook\\Python\\data bases')

train = pd.read_csv('train.csv',sep=',')
test = pd.read_csv('test.csv',sep=',')


# In[ ]:


# Seeing train dataset
train.head()


# In[ ]:


# Seeing test dataset
test.head()


# In[ ]:


# Seeing shape of each dataset
print('train shape:',train.shape)
print('teset shape:',test.shape)


# In[ ]:


# Saving survived column from train dataset to be used latter
survived = train['Survived']
train = train.drop('Survived',axis=1)


# In[ ]:


# Concatenating train and test into titanic
titanic = pd.concat([train,test],axis=0,sort=False)


# In[ ]:


# Seeing result
titanic


# ## Exploratory Data Analysis

# In[ ]:


# Ploting missing values percentage in each dataset
plt.subplots(0,0, figsize = (18,5))
ax = (titanic.isnull().sum()/len(titanic)).sort_values(ascending = False).plot.bar(color = 'blue')
plt.axhline(y=0.1, color='r', linestyle='-')
plt.title('Missing values percent per columns', fontsize = 20)

# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_height())

# set individual bar lables using above list
total = sum(totals)

# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()-.03, i.get_height()+.05,             str(round((i.get_height()/total)*100, 2))+'%', fontsize=10,
                color='black')


# In[ ]:


# Seeing passenger age distribution
plt.subplots(0,0, figsize = (15,5))
plt.plot(titanic['Age'])
plt.axhline(y=np.mean(titanic['Age']), color='r', linestyle='--') # Ploting a line with average age
plt.title('Passengers Age',fontsize=20) # Ploting the title
plt.ylabel("Ages",fontsize=13)
plt.legend(('Age','Average Age'))


# In[ ]:


# Seeing passenger Fare distribution
plt.subplots(0,0, figsize = (15,5))
plt.plot(titanic['Fare'])
plt.axhline(y=np.mean(titanic['Fare']), color='r', linestyle='--')# Ploting a line with average fare
plt.title('Passengers Fare',fontsize=20) # Ploting the title
plt.ylabel("Fare",fontsize=13)
plt.legend(('Fare','Average Fare'))


# In[ ]:


# Seeing a summary of numeric columns
titanic.describe()


# In[ ]:


# Taking a looking into Ticket colum
titanic['Ticket'].describe()


# We have 929 unique values in Ticket, let's just droop this column latter

# ## Dealing with missing values and treating variables

# In[ ]:


# let's replace na's of Age and Fare using a linear regression to predict missing age and fare
# let's replace na's from Embarked variables by 'U' to indicate an unknown value

# replacing na's of Age using linear regression method
titanic['Age'] = titanic['Age'].interpolate(method="linear",
                                         limit_direction="forward")

# replacing na's of Age using linear regression method
titanic['Fare'] = titanic['Fare'].interpolate(method="linear",
                                         limit_direction="forward")

# replacing na's of Embarked from U
titanic['Embarked'] = titanic['Embarked'].fillna('U')
titanic['Embarked'] = titanic['Embarked'].astype('category') # Converting into categorys


# In[ ]:


# Converting Ages into groups
data = [titanic]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[dataset['Age'] <=11,'Age'] =0
    dataset.loc[(dataset['Age'] >11) & (dataset['Age']<=18),'Age'] = 1
    dataset.loc[(dataset['Age'] >18) & (dataset['Age']<=22),'Age'] = 2
    dataset.loc[(dataset['Age'] >22) & (dataset['Age']<=27),'Age'] = 3
    dataset.loc[(dataset['Age'] >27) & (dataset['Age']<=33),'Age'] = 4
    dataset.loc[(dataset['Age'] >33) & (dataset['Age']<=40),'Age'] = 5
    dataset.loc[(dataset['Age'] >40) & (dataset['Age']<=66),'Age'] = 6
    dataset.loc[dataset['Age']>66,'Age'] = 6


# In[ ]:


# Converting Fares into groups
data = [titanic]
for dataset in data:
    dataset['Fare'] = dataset['Fare'].astype(int)
    dataset.loc[dataset['Fare'] <=50,'Fare'] =0
    dataset.loc[(dataset['Fare'] >50) & (dataset['Fare']<=100),'Fare'] = 1
    dataset.loc[(dataset['Fare'] >100) & (dataset['Fare']<=200),'Fare'] = 2
    dataset.loc[(dataset['Fare'] >200) & (dataset['Fare']<=300),'Fare'] = 3
    dataset.loc[(dataset['Fare'] >300) & (dataset['Fare']<=400),'Fare'] = 4
    dataset.loc[(dataset['Fare'] >400) & (dataset['Fare']<=500),'Fare'] = 5
    dataset.loc[dataset['Fare']>500,'Fare'] = 5


# In[ ]:


# Verifying Na's
titanic.isna().sum()


# Let's deal with Cabin column latter

# In[ ]:


# Converting sex into numeric
gender = {'male':0,'female':1}
data = [titanic]

for dataset in data:
    dataset['Sex'] =dataset['Sex'].map(gender)


# In[ ]:


# we have 681 unique values in Ticket variable, let's drop Ticket column from our datasets
titanic = titanic.drop('Ticket',axis=1)


# ## Creating new features

# In[ ]:


# creating a new feature using Name column
data = [titanic]

for dataset in data:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.',expand = False) # extracting titles from Name column
    
    #Replace title with more common one
    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr', 
                                                'Major','Rev','Sir','Jonkheer','Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
    dataset['Title'] = dataset['Title'].fillna(0) # fill na's
    dataset['Title']  = dataset['Title'].astype('category') # converting into categorys


# In[ ]:


# Dropping Name column from datasets
titanic = titanic.drop('Name',axis=1)


# In[ ]:


# Verifying result
titanic.head()


# In[ ]:


# Creating a new feature called "Deck" using "Cabin"
data = [titanic]

for dataset in data: # loop to create the new feature
    dataset['Cabin'] = dataset['Cabin'].fillna('H0') # replacing NA's
    dataset['Deck']  = dataset['Cabin'].map(lambda x: re.compile('([a-z,A-Z]+)').search(x).group())
    dataset['Deck']  = dataset['Deck'].fillna(0) # replacing NA's
    dataset['Deck']  = dataset['Deck'].astype('category') # converting Deck feature into categorys


# In[ ]:


# Dropping Cabin column
titanic = titanic.drop(['Cabin'],axis = 1)


# In[ ]:


# Verifying result
titanic.head()


# In[ ]:


# creating a variables for passengers that have a family aboard
titanic['family_aboard'] = np.where((titanic['Parch']>=1) & (titanic['SibSp']>=1),1,0)

# creating a variable for passengers that are single dad's
titanic['single_dad'] = np.where((titanic['Parch']>=1) & (titanic['SibSp']==0) & (titanic['Sex']==0),1,0)

# creating a variable for passengers that are single mom's
titanic['single_mom'] = np.where((titanic['Parch']>=1) & (titanic['SibSp']==0) & (titanic['Sex']==1),1,0)

# creating a variables to discribe the family size
titanic['family_small'] = np.where((titanic['Parch']<2) & (titanic['SibSp']<2),1,0)
titanic['family_median'] = np.where((titanic['Parch']==3) & (titanic['SibSp']>2),1,0)
titanic['family_large'] = np.where((titanic['Parch']>3) & (titanic['SibSp']>3),1,0)


# In[ ]:


# drooping the columns that was used to creat new features
titanic = titanic.drop(['Parch','SibSp'],axis=1)


# In[ ]:


# Verifying result
titanic.head()


# ## Dealing With Categorical Features and Spliting Titanic dataset

# In[ ]:


# converting our categorical variables into new binary variables
df_emb = pd.get_dummies(titanic['Embarked'])
df_Tit = pd.get_dummies(titanic['Title'])
df_Dec = pd.get_dummies(titanic['Deck'])

# Concat new columns with binary values into train dataset
titanic = pd.concat([titanic, df_emb, df_Tit, df_Dec], axis=1)


# In[ ]:


# drooping the columns with categorys
titanic = titanic.drop(['Embarked','Title','Deck'],axis=1)


# In[ ]:


# Verifying result
titanic.head()


# In[ ]:


# Spliting titanic dataset into train and test again
train = titanic[0:891]
test = titanic[891:1310]


# In[ ]:


# Seeing shape
print('train shape:',train.shape)
print('test shape:',test.shape)


# In[ ]:


# Saving PassengerId of test dataset to create submission dataset
passengerId = test['PassengerId']


# In[ ]:


# Drooping PassengerId column from train and test
train = train.drop('PassengerId',axis=1)
test = test.drop('PassengerId',axis=1)


# In[ ]:


# Seeing train dataset first5 rows
train.head()


# In[ ]:


# Seeing test dataset first5 rows
test.head()


# ## Creating Models and Make Predictions

# In[ ]:


# Separing datasets to training and prediction
x_train = train
y_train = survived
x_test = test


# ###### Decision Tree Pruned by 5 max_depth to trying to avoid overfitting

# In[ ]:


# Creating a Decision Tree Classifier model
tree = DecisionTreeClassifier(criterion = "gini", max_depth = 5)

# Training the model
tree.fit(x_train,y_train)


# In[ ]:


# Applying our model in train dataset
y_pred_train = tree.predict(x_train)


# In[ ]:


# Seeing results
print(confusion_matrix(y_train, y_pred_train))
print()
print("--------------------------------------------------------------------------------")
print()
print(classification_report(y_train, y_pred_train))


# In[ ]:


print("Accuracy:", round(metrics.accuracy_score(y_train, y_pred_train),3)*100)


# In[ ]:


# Applying our model in test dataset
y_pred_test = tree.predict(x_test)


# In[ ]:


# Creating the dataset to submission
submission = pd.DataFrame({"PassengerId":passengerId,"Survived":y_pred_test})
submission.to_csv('DecisionTree.csv',index = False)


# In[ ]:


# Plooting our tree
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

dot_data = StringIO() #open the input and output from jupyter
# to external program

#creating the plot
export_graphviz(tree,
               out_file = dot_data,
               filled = True,
               rounded = True,
               special_characters = True,
               feature_names = x_train.columns[0:30],
               class_names = ['0','1'])

#pushing the plot from external program to jupyter
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

#ploting the tree
Image(graph.create_png())


# ###### Let's now submiss our prediction in test dataset to kaggle competition

# In[ ]:


# Seeing results in test dataset submited in kaggle/ importing image of results from our directory
Image(filename='decision tree 1.PNG')


# ###### The result was not soo good, 84,6% in train but only 77,03 % in test, our model are not capable of generalize let's try to improve this results pruning our tree again

# In[ ]:


# Creating a Decision Tree Classifier model pruning by 3 max_depth
tree = DecisionTreeClassifier(criterion = "gini", max_depth = 3)

# Training the model
tree.fit(x_train,y_train)

# Applying our model in train dataset
y_pred_train = tree.predict(x_train)

# Seeing results
print(confusion_matrix(y_train, y_pred_train))
print()
print("--------------------------------------------------------------------------------")
print()
print(classification_report(y_train, y_pred_train))


# In[ ]:


print("Accuracy:", round(metrics.accuracy_score(y_train, y_pred_train),3)*100)


# In[ ]:


# Applying our model in test dataset
y_pred_test = tree.predict(x_test)

# Creating the dataset to submission
submission = pd.DataFrame({"PassengerId":passengerId,"Survived":y_pred_test})
submission.to_csv('DecisionTree2.csv',index = False)


# In[ ]:


# Plooting our tree
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

dot_data = StringIO() #open the input and output from jupyter
# to external program

#creating the plot
export_graphviz(tree,
               out_file = dot_data,
               filled = True,
               rounded = True,
               special_characters = True,
               feature_names = x_train.columns[0:30],
               class_names = ['0','1'])

#pushing the plot from external program to jupyter
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

#ploting the tree
Image(graph.create_png())


# ###### Let's submiss our prediction again

# In[ ]:


# Seeing results in test dataset submited in kaggle/ importing image of results from our directory
Image(filename='decision tree 2.PNG')


# ###### 80% of accuracy, a great result!, 82,5% in train and 80,38% in test dataset, the model now are capable of generalize

# In[ ]:




