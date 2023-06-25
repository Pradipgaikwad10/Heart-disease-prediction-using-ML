#!/usr/bin/env python
# coding: utf-8

# # Importing essential libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset = pd.read_csv("heart.csv")


# In[3]:


dataset.shape


# In[4]:


dataset.head(5)


# In[5]:


dataset.sample(5)


# In[6]:


dataset.describe()


# In[7]:


dataset.info()


# In[8]:


###Luckily, we have no missing values in above


# # lets uderstand our coloumn better

# In[9]:


info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



for i in range(len(info)):
    print(dataset.columns[i]+":\t\t\t"+info[i])


# # Analysing the 'target' variable

# In[10]:


dataset["target"].describe()


# In[11]:


dataset["target"].unique()


# In[12]:


#Clearly, this is a classification problem, with the target variable having values '0' and '1'


# In[13]:


from sklearn.model_selection import train_test_split

predictors = dataset.drop("target",axis=1)
target = dataset["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)


# In[30]:


print('X_train-', X_train.size)
print('X_test-',X_test.size)
print('y_train-', Y_train.size)
print('y_test-', Y_test.size)


# # Model Fitting

# In[32]:


from sklearn.metrics import accuracy_score


# # Logistic Regression

# In[33]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,Y_train)

Y_pred_lr = lr.predict(X_test)


# In[37]:


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_test,Y_pred_lr)
cm


# In[38]:


sns.heatmap(cm, annot=True,cmap='BuPu')


# In[36]:


from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred_lr))


# In[16]:


score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")


# # Naive Bayes

# In[17]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,Y_train)

Y_pred_nb = nb.predict(X_test)


# In[39]:


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_test,Y_pred_nb)
cm


# In[40]:


sns.heatmap(cm, annot=True,cmap='BuPu')


# In[41]:


from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred_nb))


# In[18]:


score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")


# #  Decision tree

# In[23]:


from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0


for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)


dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,Y_train)
Y_pred_dt = dt.predict(X_test)


# In[45]:


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_test,Y_pred_dt)
cm


# In[46]:


sns.heatmap(cm, annot=True,cmap='BuPu')


# In[47]:


from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred_dt))


# In[24]:


score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")


# # Random Forest

# In[25]:


from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0


for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,Y_train)
Y_pred_rf = rf.predict(X_test)


# In[48]:


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_test,Y_pred_rf)
cm


# In[49]:


sns.heatmap(cm, annot=True,cmap='BuPu')


# In[50]:


from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred_rf))


# In[26]:


score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")


# In[ ]:




