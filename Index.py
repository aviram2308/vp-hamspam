
# coding: utf-8

# # Ham Spam Filter for Mobile Phone Messsages

# In[2]:


# import the dataset containing messages along with their ham spam labels
import pandas as pd
df=pd.read_csv("https://cdn.rawgit.com/aviram2308/vp-hamspam/af94d24e/spam%20(1).csv", encoding="ISO-8859-1")


# In[3]:


df.head()


# In[4]:


df=df.iloc[:,0:2]
df.head()


# In[5]:


df=df.rename(columns={'v1':'Label', 'v2':'Message'})
df.head()


# In[6]:


# converting character categorical to numeric categorical
df['Label_num'] = df.Label.map({'ham':0, 'spam':1})
df.head(10)


# In[7]:


#extracting feature matrix and response vector
X=df.Message
y=df.Label_num


# In[8]:


# split X and y into training and testing sets
from sklearn import model_selection as ms
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, random_state=1)


# In[9]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[10]:


# import and instantiate CountVectorizer (with the default parameters)# impor 
from sklearn.feature_extraction.text import CountVectorizer


# In[11]:


vect = CountVectorizer(min_df=0.04,max_df=.96)


# In[12]:


# learn the 'vocabulary' of the training data (occurs in-place)
vect.fit(X_train)


# In[13]:


# transform training data into a 'document-term matrix'
dtm_train=vect.transform(X_train)
dtm_test=vect.transform(X_test)
type(dtm_train)


# In[14]:


#Comparing different models/estimators
from sklearn import linear_model as lm
est1=lm.LogisticRegression()
est1.fit(dtm_train,y_train)


# In[15]:


# checking if a given message is ham(0) or spam(1)
dtm_3=vect.transform(['Data cleansing is hard to do, hard to maintain, hard to know where to start'])
est1.predict(dtm_3)


# In[16]:


dtm_3=vect.transform(['Hello, plz do this'])
est1.predict(dtm_3)


# In[17]:


dtm_3=vect.transform(["The Facebook Team wish to inform you that you are one of the Lucky winners from your Region (PAKISTAN) in this year's Online Promotion and your Facebook account has won the sum of EIGHT HUNDRED THOUSAND GREAT BRITISH POUNDS (£800,000 GBP)"])
est1.predict(dtm_3)


# In[18]:


#calculating y predicted and y predicted probability for estimator 1
y_pred=est1.predict(dtm_test)
y_pred_prob=est1.predict_proba(dtm_test)
y_pred_prob.shape


# In[19]:


# calculating score for estimator 1(sc1) and roc score(scp1)
from sklearn import metrics as mt


# In[20]:


sc1=mt.accuracy_score(y_test,y_pred)
sc1


# In[21]:


scp1=mt.roc_auc_score(y_test,y_pred_prob[:,1])
scp1


# In[22]:


# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
est2= MultinomialNB()


# In[23]:


est2.fit(dtm_train,y_train)


# In[24]:


dtm_3=vect.transform(['Data cleansing is hard to do, hard to maintain, hard to know where to start'])
est2.predict(dtm_3)


# In[25]:


dtm_3=vect.transform(['Hello, plz do this'])
est2.predict(dtm_3)


# In[26]:


dtm_3=vect.transform(["The Facebook Team wish to inform you that you are one of the Lucky winners from your Region (PAKISTAN) in this year's Online Promotion and your Facebook account has won the sum of EIGHT HUNDRED THOUSAND GREAT BRITISH POUNDS (£800,000 GBP)"])
est2.predict(dtm_3)


# In[27]:


y_pred2=est2.predict(dtm_test)
y_pred_prob2=est2.predict_proba(dtm_test)
y_pred_prob2.shape


# In[28]:


from sklearn import metrics as mt
sc2=mt.accuracy_score(y_test,y_pred2)
sc2


# In[29]:


scp2=mt.roc_auc_score(y_test,y_pred_prob2[:,1])
scp2


# In[30]:


# instantiate random forest model with default paramter
from sklearn import ensemble as eb
est3=eb.RandomForestClassifier(random_state=5)


# In[31]:


est3.fit(dtm_train,y_train)


# In[32]:


dtm_3=vect.transform(['Data cleansing is hard to do, hard to maintain, hard to know where to start'])
est3.predict(dtm_3)


# In[33]:


dtm_3=vect.transform(['Hello, plz do this'])
est3.predict(dtm_3)


# In[34]:


dtm_3=vect.transform(["The Facebook Team wish to inform you that you are one of the Lucky winners from your Region (PAKISTAN) in this year's Online Promotion and your Facebook account has won the sum of EIGHT HUNDRED THOUSAND GREAT BRITISH POUNDS (£800,000 GBP)"])
est3.predict(dtm_3)


# In[35]:


y_pred3=est3.predict(dtm_test)
y_pred_prob3=est3.predict_proba(dtm_test)
y_pred_prob3.shape


# In[36]:


from sklearn import metrics as mt
sc3=mt.accuracy_score(y_test,y_pred3)
sc3


# In[37]:


scp3=mt.roc_auc_score(y_test,y_pred_prob3[:,1])
scp3


# In[38]:


# from calculated scores, we select random forest model
# To tune parameters of random forest model,  make parameter grid(pg) 
n=[5,10,20,40]
c=['gini', 'entropy']
m=[5,20,25,30]
m2=[.2,.1,.05]


# In[39]:


pg=dict(n_estimators=n, criterion=c, max_features=m)


# In[40]:


#make a score grid using random search cross validation
#grid search cross validation does exhaustive search and takes a lot of computational resource
#first score grid is made using 10 iterations, thus, sg10
sg10=ms.RandomizedSearchCV(est3,pg,cv=10,scoring='accuracy', n_iter=10,random_state=5)


# In[41]:


dtm=vect.transform(X)
sg10.fit(dtm,y)


# In[42]:


sg10.best_score_


# In[43]:


sg20=ms.RandomizedSearchCV(est3,pg,cv=10,scoring='accuracy', n_iter=20,random_state=5)


# In[44]:


sg20.fit(dtm,y)


# In[45]:


sg20.best_score_


# In[46]:


sg25=ms.RandomizedSearchCV(est3,pg,cv=10,scoring='accuracy', n_iter=25,random_state=5)


# In[47]:


sg25.fit(dtm,y)


# In[48]:


sg25.best_score_


# In[49]:


sg20.best_params_


# In[50]:


sg25.best_params_


# In[51]:


# creating a model with tuned parameters
est4=eb.RandomForestClassifier(criterion= 'entropy', max_features= 5, n_estimators= 40, random_state=5)


# In[52]:


est4.fit(dtm_train,y_train)


# In[53]:


y_pred4=est4.predict(dtm_test)
y_pred_prob4=est4.predict_proba(dtm_test)


# In[54]:


sc4=mt.accuracy_score(y_test,y_pred4)
sc4


# In[55]:


scp4=mt.roc_auc_score(y_test,y_pred_prob4[:,1])
scp4


# In[58]:


# calculating sensitivities of all models
print(mt.recall_score(y_test,y_pred))
print(mt.recall_score(y_test,y_pred2))
print(mt.recall_score(y_test,y_pred3))
print(mt.recall_score(y_test,y_pred4))


# In[59]:


# for spam detection type 1 error more important than type 2
# therefore specificity more important than sensitivity
#create confusion matrix for all models
cm=mt.confusion_matrix(y_test,y_pred)

cm2=mt.confusion_matrix(y_test,y_pred2)

cm3=mt.confusion_matrix(y_test,y_pred3)

cm4=mt.confusion_matrix(y_test,y_pred4)


# In[61]:


# calculating specificity for all models using confusion matrix of the respective model
sp1=cm[0,0]/(cm[0,0]+cm[0,1])
print(sp1)
sp2=cm2[0,0]/(cm2[0,0]+cm2[0,1])
print(sp2)
sp3=cm3[0,0]/(cm3[0,0]+cm3[0,1])
print(sp3)
sp4=cm4[0,0]/(cm4[0,0]+cm4[0,1])
print(sp4)



# In[65]:


se4=mt.recall_score(y_test,y_pred4)


# In[66]:


# choosing estimator 4 with following optimal metrics
print('Accuracy score: ',sc4)
print('ROC AUC score: ',scp4)
print('Sensitivity: ',se4)
print('Specificity: ',sp4)

