#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Using Chess (King-Rook vs. King) Data Set  from:
# https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King%29
# to create a model to classify: optimal depth-of-win for White in 0 to 16 moves


# In[2]:


# Charging librarys initial librarys
import pandas as pd
import numpy as np
import os


# In[3]:


# Importing dataset from UCI machine learning repository passing columns names and delimiter
xadrez = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data',
                    delimiter=",",names=list(['White King file(column)','White King rank(row)',
                                              'White Rook file','White Rook rank','Black King file',
                                              'Black King rank','class']))
xadrez.head()


# In[4]:


# Converting our variables that are not numeric values into categoricals
xadrez["White King file(column)"] = xadrez['White King file(column)'].astype('category')
xadrez["White Rook file"] = xadrez['White Rook file'].astype('category')
xadrez["Black King file"] = xadrez['Black King file'].astype('category')
xadrez["class"] = xadrez['class'].astype('category')

xadrez.info()


# In[5]:


# converting our categorical variables into new binary variables
df_wkf = pd.get_dummies(xadrez['White King file(column)'])
df_wrf = pd.get_dummies(xadrez['White Rook file'])
df_bkf = pd.get_dummies(xadrez['Black King file'])

# Concat new columns with binary values into xadrez dataset
xadrez = pd.concat([xadrez, df_wrf, df_bkf, df_wkf], axis=1)

xadrez.head()


# In[6]:


# Removing the old columns that are converted into binary
xadrez = xadrez.drop(['White King file(column)','White Rook file','Black King file'],axis=1)

xadrez.head()


# In[7]:


# Visualysing catecories of our class
print(xadrez['class'].cat.categories)


# In[8]:


# Converting each class into new individual binary classes
df_class = pd.get_dummies(xadrez['class'])

df_class.head()


# In[9]:


# -----------------------------------------------------------------------#
# ------------ Creating a ensemble model using DecisionTreeClassifier ---#
#------------- to train and predict each binary class -------------------#
# ------------ ----------------------------------------------------------#


# In[10]:


# Creating two variables to calculate average accuracy
accuracy = 0
m=0

# Creating a loop to train and predict each class
for column in df_class:
    
    x = xadrez.drop('class',axis=1) #separating varibles into x
    y = df_class[column] #separating target(class) into y

    #importing train_test_split function to split dataset into train and test
    from sklearn.model_selection import train_test_split

    #spliting dataset
    np.random.seed(42)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3) #70% to train and 30% to test

    #importing DecisionTreeClassifier function to train the model
    from sklearn.tree import DecisionTreeClassifier

    #creating the model and storage into tree varible
    tree = DecisionTreeClassifier()

    #training the model
    tree.fit(x_train,y_train)

    #applying our model in test dataset
    y_pred_test = tree.predict(x_test)
    
    #geting the last prediction
    df = pd.DataFrame(y_pred_test,columns =[column])
    df = df.tail(1)
    
    # If the class are the 'draw' then we create a DataFrame to storage
    # the next last prediction of another classes
    if column=='draw':
        pred = pd.DataFrame(0,columns=['test'],index=[df.index[0]])
    
    # Concat the last prediction with pred Dataframe
    pred = pd.concat([pred, df], axis=1)
    
    #--------------------------------------------------------------#
    #--------- Seeing results of each prediction separately -------#
    #--------------------------------------------------------------#

    # Importing 'classification_report' and 'confusion_matrix' functions
    from sklearn.metrics import classification_report, confusion_matrix

    print("confusion matrix of:",column)
    print(confusion_matrix(y_test,y_pred_test)) #printing confusion matrix
    print()
    print()
    print(classification_report(y_test,y_pred_test)) #printing classification report

    # Importing metrics to calculate precision/accuracy
    import sklearn.metrics as metrics

    #calculating and printing precision
    print("Accuracy:", round(metrics.precision_score(y_test,y_pred_test),3))
    print()
    print("------------------------------------------------") #given a space
    print()
    
    # Sum all accuracys of each prediction to calculate average accuracy
    accuracy = accuracy + metrics.precision_score(y_test,y_pred_test)
    m=m+1

# Calculating average accuracy
accuracy = accuracy/m

# Ploting results
print('Prediction of each class:')
print()
print(pred.head())


# In[15]:


# Printing average accuracy
print('Average accuracy:',round(accuracy*100,2),'%')


# In[16]:


# Selecting dataset xadrez with the classes
full_data = xadrez.drop('class',axis=1)

# Selecting a new row of data to realize the prediction
data_to_prediction = full_data[5000:5001] #selecting the row 5000
data_to_prediction.head()


# In[17]:


# Now we can execute almost the same lines of code in model production
accuracy = 0
m=0

for column in df_class:
    
    x = xadrez.drop('class',axis=1) #separing the varibles into x
    y = df_class[column] #separing the target(class) into y

    from sklearn.model_selection import train_test_split

    # Spliting dataset
    np.random.seed(42)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

    # Importing DecisionTreeClassifier function to train the model
    from sklearn.tree import DecisionTreeClassifier

    # Creating the model and storage into tree varible
    tree = DecisionTreeClassifier()

    # Training the model
    tree.fit(x_train,y_train)
    
    # Recieve new data to predict
    x_test = data_to_prediction

    # Applying our model in test dataset
    y_pred_test = tree.predict(x_test)
    
    # Geting the last prediction
    df = pd.DataFrame(y_pred_test,columns =[column])
    df = df.tail(1)
    
    # If the class are the 'draw' then we create a DataFrame to storage
    # the next last prediction of another classes
    if column=='draw':
        pred = pd.DataFrame(0,columns=['test'],index=[df.index[0]])
    
    # Concat the last prediction with pred Dataframe
    pred = pd.concat([pred, df], axis=1)

# Ploting a DataFrame with each prediction
print('Prediction of each class:')
print()
print(pred.head())


# In[21]:


# Ploting class that was predicted to the last row
for i in pred.columns:
    newdf = pred.loc[lambda pred: pred[i] == 1]
    if(not newdf.empty):
        print('The optimal depth-of-win moves for White is:')
        print()
        print(i)


# In[22]:


# Verifying if the class predicted was right
xadrez[5000:5001]


# In[ ]:




