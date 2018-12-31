
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from  pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import Imputer


# In[2]:


Train = pd.read_csv("C:/Users/Adi/Desktop/project/Train_data.csv")
Test = pd.read_csv("C:/Users/Adi/Desktop/project/Test_data.csv")


# In[3]:


# COMBINING TRAIN AND TEST DATA
Train["ID"] = 1
Test["ID"] = 2
Churn_data = Train.append(Test)
Churn_data = Churn_data.reset_index(drop=True)
print(Churn_data.shape)
Churn_data.head(5000)

Churn_data.Churn = np.where(Churn_data.Churn==' False.', 0, 1)

# for column 'voice mail plan'
Churn_data['voice mail plan'] = Churn_data['voice mail plan'].astype('category').cat.codes
# for column 'voice mail plan'
Churn_data['international plan'] = Churn_data['international plan'].astype('category').cat.codes
# for column 'state'
Churn_data.state = Churn_data.state.astype('category').cat.codes

Churn_data = Churn_data.drop(labels=['phone number'], axis=1)     #dropping phone number column
Churn_data.Churn = Churn_data.Churn.astype('object')

corr = Churn_data.corr()
sns.heatmap(corr)


# In[4]:


p_values = []
for col in Churn_data.columns:
    print(col)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(Churn_data.Churn, Churn_data[col]))
    p_values.append(p)
    print(p)


# In[5]:


# Selecting features based on the above two results
selected_features = []
for val in p_values:
    selected_features.append(val<0.05)
    
Churn_data = Churn_data.iloc[:, selected_features]
Churn_data = Churn_data.drop(labels=['total day minutes', 'total intl minutes'], axis=1)
Churn_data.shape

Churn_data['ID'] = Train.ID
Churn_data.ID = np.where(Churn_data.ID==1, 1, 0)

cols = ['total day charge','total eve charge', 'total intl charge', 'number customer service calls']
plt.figure()
i = 1
plt.suptitle('Before Normalization', x=0.5, y=1.05, ha='center', fontsize='xx-large')
for col in cols:
    plt.subplot(1,4,i)
    p = Churn_data[col].diff().hist(figsize=(15,3), bins=30)
    p.set_xlabel(col)
    i = i + 1
    
# cols = ['total day charge','total eve charge', 'total intl charge', 'number customer service calls']
plt.figure()
plt.suptitle('After Normalization', x=0.5, y=1.05, ha='center', fontsize='xx-large')
i = 1
for col in cols:
    plt.subplot(1,4,i)
    Churn_data[col] = (Churn_data[col] - Churn_data[col].min())/(Churn_data[col].max() - Churn_data[col].min())
    p = Churn_data[col].diff().hist(figsize=(15,3), bins=30, color = 'g')
    p.set_xlabel(col)
    i = i + 1

cols = ['total day charge','total eve charge', 'total intl charge', 'number customer service calls']
plt.figure()
i = 1
plt.suptitle('Before Standardization', x=0.5, y=1.05, ha='center', fontsize='xx-large')
for col in cols:
    plt.subplot(1,4,i)
    p = Churn_data[col].diff().hist(figsize=(15,3), bins=30)
    p.set_xlabel(col)
    i = i + 1


# In[6]:


# cols = ['total day charge','total eve charge', 'total intl charge', 'number customer service calls']
plt.figure()
plt.suptitle('After Standardization', x=0.5, y=1.05, ha='center', fontsize='xx-large')
i = 1
for col in cols:
    plt.subplot(1,4,i)
    Churn_data[col] = (Churn_data[col] - Churn_data[col].mean()) / Churn_data[col].std()
    p = Churn_data[col].diff().hist(figsize=(15,3), bins=30, color = 'y')
    p.set_xlabel(col)
    i = i + 1


# In[7]:


train = Churn_data[Churn_data.ID == 1]
test = Churn_data[Churn_data.ID == 0]
train = train.drop(labels='ID', axis=1)
test = test.drop(labels='ID', axis=1)
print("Shape of train : " + str(train.shape))
print("Shape of test : " + str(test.shape))

train = train.drop(labels='state', axis=1)
test = test.drop(labels='state', axis=1)

mod_name = []
accuracy = []
precision_score = []
recall_score = []
f1_score = []
false_negative = []

def register_model_with_scores(name, acc, precision, recall, f1, fn):
    mod_name.append(name)
    accuracy.append(acc)
    precision_score.append(precision)
    recall_score.append(recall)
    f1_score.append(f1)
    false_negative.append(fn)

    
def generate_score(model_name, test, predicted, print_scores = 1, register_model = 1):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    acc = accuracy_score(test.Churn.astype('float'), predicted)
    f1 = f1_score(test.Churn.astype('float'), predicted)
    precision = precision_score(test.Churn.astype('float'), predicted)
    recall = recall_score(test.Churn.astype('float'), predicted)
    conf_mat = pd.crosstab(test.Churn, predicted, rownames=['actual'], colnames=['predicted'])
    FN = conf_mat.iloc[1,0]
    if(print_scores==1):
        print("Accuracy : " + str(acc))
        print("Precision : " + str(precision))
        print("Recall : " + str(recall))
        print("False Negative : " + str(FN))
        print("F1 : " + str(f1))
    if(register_model==1):    
        register_model_with_scores(model_name, acc, precision, recall, f1, FN)
    return conf_mat, f1

except_Churn = train.columns
except_Churn = except_Churn[0:8]
except_Churn

from sklearn import tree
model_name = 'Decision Tree'
dt = tree.DecisionTreeClassifier(criterion='entropy')
dt.fit(X=train[except_Churn].astype('float'), y=train.Churn.astype('float'))

predicted = dt.predict(test[except_Churn])
predicted = np.where(predicted>0.5, 1, 0)
cm,f1 = generate_score(model_name, test, predicted)
cm


# In[8]:


#RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
model_name = 'Random Forest'
rf = RandomForestClassifier(random_state=100, n_jobs=4, criterion='entropy')
rf.fit(X=train[except_Churn].astype('float'), y=train.Churn.astype('float'))

predicted = rf.predict(test[except_Churn])
predicted = np.where(predicted>0.5, 1, 0)
cm,f1 = generate_score(model_name, test, predicted, register_model=0)
cm

#  best n_estimator value
f1_for_n_estimators = []
for i in range(1,50):
    rf = RandomForestClassifier(random_state=100, n_jobs=4, criterion='entropy', n_estimators=i)
    rf.fit(X=train[except_Churn].astype('float'), y=train.Churn.astype('float'))
    predicted = rf.predict(test[except_Churn])
    predicted = np.where(predicted>0.5, 1, 0)
    cm,f1 = generate_score(model_name, test, predicted, print_scores=0, register_model=0)
    f1_for_n_estimators.append(f1)
    
# print(f1_for_n_estimators)    
count = 1
max_val = max(f1_for_n_estimators)
for i in f1_for_n_estimators:
    if (i == max_val):
        best_estimator = count
    count = count + 1    
print("Best estimator value : " + str(best_estimator))

rf = RandomForestClassifier(random_state=100, n_jobs=4, criterion='entropy', n_estimators=45)
rf.fit(X=train[except_Churn].astype('float'), y=train.Churn.astype('float'))
predicted = rf.predict(test[except_Churn])
predicted = np.where(predicted>0.5, 1, 0)
cm,f1 = generate_score(model_name, test, predicted)
cm

from sklearn.naive_bayes import GaussianNB
model_name = 'Naive Bayes'
gnb = GaussianNB()
gnb.fit(X=train[except_Churn].astype('float'), y=train.Churn.astype('float'))

predicted = gnb.predict(test[except_Churn])
predicted = np.where(predicted>0.5, 1, 0)
cm,f1 = generate_score(model_name, test, predicted)
cm

df = pd.DataFrame({'model_name' : mod_name, 
              'accuracy' : accuracy, 
              'precision_score' : precision_score, 
              'recall_score' : recall_score, 
              'f1_score' : f1_score, 
              'false_negative' : false_negative}, 
                  columns=['model_name', 'precision_score', 'recall_score', 'false_negative', 'accuracy', 'f1_score'])
df

