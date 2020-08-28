#!/usr/bin/env python
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from fancyimpute import KNN as fn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
import seaborn as sns


# In[3]:


tr_data = pd.read_excel("Project - 4 - Train Data.xlsx")


# In[446]:


tr_data


# In[447]:


tr_data.isnull().sum()


# In[448]:


train_data = tr_data.drop(['Loanapp_ID','first_name','last_name','email','address', 'INT_ID','Prev_ID','AGT_ID'],axis=1)


# In[449]:


print(train_data.dtypes)


# In[450]:


train_data_array = train_data.values
train_data_array


# # Using simple imputer and KNN imputer for categorical and numerical data respectively

# In[451]:


imp = SimpleImputer(strategy="most_frequent")


# In[452]:


le = LabelEncoder()


# In[453]:


categorical_feature_mask = train_data.dtypes==object
categorical_columns = train_data.columns[categorical_feature_mask].tolist()
categorical_columns


# In[454]:


numerical_feature_mask = train_data.dtypes==float
numerical_columns = train_data.columns[numerical_feature_mask].tolist()
numerical_columns


# In[455]:


train_cat = train_data[categorical_columns]
train_num = train_data[numerical_columns]


# In[456]:


train_cat_data = train_data.drop(columns=['Dependents',
 'App_Income_1',
 'App_Income_2',
 'CPL_Amount',
 'CPL_Term',
 'Credit_His'])


# In[457]:


train_cat = imp.fit_transform(train_cat_data)
train_cat


# In[458]:


train_num = fn(k=3).fit_transform(train_num) #knn_imputer
train_num


# In[459]:


train_num.shape


# In[460]:


train_cat.shape


# In[461]:


train_cat_df = pd.DataFrame(data = train_cat[:,:],columns = categorical_columns)


# In[462]:


train_num_df = pd.DataFrame(data = train_num[:,:],columns = numerical_columns)


# In[463]:


trainf_data = pd.concat([train_cat_df,train_num_df],axis='columns')


# In[464]:


trainf_data


# In[465]:


cpl_status_df = trainf_data["CPL_Status"]


# In[466]:


declined_number=trainf_data[trainf_data['CPL_Status']=='N'].count()
approved_number = trainf_data[trainf_data['CPL_Status']=='Y'].count()


# In[467]:


declined_percentage = declined_number/614*100
approved_percentage = approved_number/614
print(f"declined_percentage:",declined_percentage)


# In[ ]:





# In[468]:


trainf_data.isnull().sum()


# In[469]:


to_encode  = trainf_data[categorical_columns].astype('category')


# In[470]:


to_encode


# In[471]:


to_encode


# # These are columns we would be using for prediction

# In[472]:


checking_columns = ['Sex', 'Marital_Status', 'Qual_var', 'SE', 'Prop_Area','Dependents','App_Income_1','App_Income_2',
                    'CPL_Amount','CPL_Term','Credit_His']


# In[473]:


non_category_columns = numerical_columns


# In[474]:


all_encode = pd.concat([to_encode,trainf_data[non_category_columns]],axis='columns')


# In[475]:


all_encode


# In[476]:


all_encode.columns


# In[477]:


all_encode.isnull().sum()


# In[478]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer


# # applying one hot encoding on categorical data and scaling to numerical data

# In[479]:


dummies_all_encode = pd.get_dummies(all_encode,drop_first = True)


# In[480]:


dummies_all_encode


# In[481]:


dummies_all_encode_1 = dummies_all_encode.drop(columns='CPL_Status_Y') #encoded x_data


# In[482]:


depending_columns = dummies_all_encode_1.columns #encoded columns
depending_columns


# In[483]:


ct = ColumnTransformer([("standard scaling", StandardScaler(),[0,1,2,3,4,5])],remainder = 'passthrough')


# In[484]:


X_train= ct.fit_transform(dummies_all_encode_1)
X_train = X_train.astype('float') #checking if every value is fl


# In[485]:


#X_train = np.delete(X_train,12,axis = 1)


# In[486]:


onehot_df = pd.DataFrame(X_train,columns=depending_columns)


# In[487]:


X_train_df = onehot_df


# In[488]:


onehot_df #scaled values


# In[489]:


onehot_df_with_status = pd.concat([onehot_df,dummies_all_encode["CPL_Status_Y"]],axis="columns")


# In[490]:


X_train.shape


# In[491]:


y = dummies_all_encode['CPL_Status_Y']
type(y)


# In[492]:


y_train = y


# In[493]:


y_train_df = pd.DataFrame(y_train)
y_train_df


# In[494]:


X_train.shape


# In[495]:


y_train.shape


# # Applying same for test data for checking

# In[496]:


trest_data = pd.read_excel("Project - 4 - Test Data.xlsx")


# In[497]:


trest_data


# In[498]:


len(trest_data.columns)


# In[499]:


test_data = trest_data.drop(['Loanapp_ID','first_name','last_name','email','address', 'INT_ID','Prev_ID','AGT_ID'],axis=1)
len(test_data.columns)


# In[500]:


print(test_data.dtypes)


# In[501]:


test_data_array = test_data.values
test_data_array


# # Using simple imputer and KNN imputer for categorical and numerical test data respectively

# In[502]:


imp = SimpleImputer(strategy="most_frequent")


# In[503]:


le = LabelEncoder()


# In[504]:


categorical_test_feature_mask = test_data.dtypes==object
categorical_test_columns = test_data.columns[categorical_test_feature_mask].tolist()
categorical_test_columns


# In[505]:


numerical_test_feature_mask = test_data.dtypes==float
numerical_test_columns = test_data.columns[numerical_test_feature_mask].tolist()
numerical_test_columns


# In[506]:


test_data[categorical_test_columns]


# In[507]:


test_cat = test_data[categorical_test_columns]
test_num = test_data[numerical_test_columns]


# In[508]:


test_cat


# In[509]:


test_cat = imp.fit_transform(test_cat)
test_cat


# In[510]:


test_num = fn(k=3).fit_transform(test_num) #knn_imputer
test_num


# In[511]:


test_num.shape


# In[512]:


test_cat.shape


# In[513]:


test_cat_df = pd.DataFrame(data = test_cat[:,:],columns = categorical_test_columns)


# In[514]:


test_num_df = pd.DataFrame(data = test_num[:,:],columns = numerical_test_columns)


# In[515]:


testf_data = pd.concat([test_cat_df,test_num_df],axis='columns')


# In[516]:


testf_data


# In[517]:


testf_data.isnull().sum()


# In[ ]:





# # These are columns we would be using for prediction

# In[518]:


checking_test_columns = ['Sex', 'Marital_Status', 'Qual_var', 'SE', 'Prop_Area','Dependents','App_Income_1','App_Income_2',
                    'CPL_Amount','CPL_Term','Credit_His']


# In[519]:


non_category_test_columns = numerical_test_columns


# In[520]:


#all_test_encode = pd.concat([to_encode_test,testf_data[non_category_test_columns]],axis='columns')


# In[521]:


testf_data


# In[522]:


testf_data.columns


# In[523]:


testf_data.isnull().sum()


# In[524]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer


# # applying one hot encoding on categorical data and scaling to numerical data

# In[525]:


test_encode = pd.get_dummies(testf_data,drop_first = True)
test_encode


# In[526]:


ct = ColumnTransformer([("standard scaling", StandardScaler(),[0,1,2,3,4,5])],remainder = 'passthrough')


# In[527]:


X_test_final = ct.fit_transform(test_encode)
X_test_final = X_test_final.astype('float') #checking if every value is float


# In[528]:


onehot_test_df = pd.DataFrame(X_test_final,columns=test_encode.columns)


# In[529]:


onehot_test_df


# In[530]:


Excel_columns = onehot_test_df.columns


# # Finding corelation between columns in dataframe

# In[531]:


type(onehot_df_with_status)


# In[532]:


corelation_data = onehot_df_with_status.corr()
corelation_data


# In[533]:


#we can see credit history has highest corelation


# In[534]:


ax = sns.heatmap(corelation_data,vmin=-1,vmax=1,center=0,cmap=sns.diverging_palette(20,220,n=200),square=True)


# # Logistic regression for prediction

# In[535]:


logreg = LogisticRegression()
logreg_model = logreg.fit(X_train, y_train) #X_train and y_train


# In[536]:


logreg_model.score(X_train,y_train)


# In[537]:


logreg_train_predicted = logreg_model.predict(X_train)


# In[538]:


logreg_accuracy = accuracy_score(y_train,logreg_train_predicted)
print(logreg_accuracy)


# In[539]:


logreg_test = logreg_model.predict(X_test_final)
logreg_test_df = pd.DataFrame(logreg_test,columns=["logreg"])


# In[540]:


logreg_test_df


# In[541]:


display_labels_CM = ['dependent_variables','CPL_status']


# In[542]:


np.set_printoptions(precision=2)
titles_options = [("LogReg confusion matrix without normalization",None),
                  ("Logreg confusion matrix with normalization",'true')]
for title,normalize in titles_options:
    disp = plot_confusion_matrix(logreg_model,X_train,y_train,
                                 display_labels=display_labels_CM,cmap=plt.cm.Oranges,normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()


# # using SVM for classification
#

# In[543]:


from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


# In[544]:


import warnings
warnings.filterwarnings('ignore')


# In[545]:


support = Pipeline((("scaler",StandardScaler()),("linear_svc",LinearSVC(C=1,loss = "hinge")),))


# In[546]:


svc_x = X_train


# In[547]:


svc_y = y_train


# In[548]:


svc_model = support.fit(svc_x,svc_y)


# # calculating accuracy score of svc

# In[549]:


svc_train_predict = support.predict(svc_x)


# In[550]:


def rounding(x): #function for rounding upto 4 digits
    return round(x,4)


# In[551]:


#svc_accuracy = accuracy_score(svc_y,svc_train_predict)
print(svc_model.score(X_train,y_train)*100)


# In[552]:


svc_test_predict = support.predict(X_test_final)


# In[553]:


svc_test_predict.reshape(-1,1)
type(svc_test_predict)


# # svc_test_predicted_dataframe

# In[554]:


svc_test_predict_df = pd.DataFrame(data = svc_test_predict.reshape(-1,1),columns=["svc_predict"])


# In[555]:


np.set_printoptions(precision=2) #making precision to 2 digits
# Plot non-normalized confusion matrix
titles_options = [("SVC Confusion matrix, without normalization", None),
                  ("SVC Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(svc_model, X_train, y_train,
                                 display_labels=display_labels_CM,
                                 cmap=plt.cm.copper,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# #   K nearest neighbours for classification

# In[556]:


X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_train,y_train,test_size =0.3)


# In[557]:


from sklearn.neighbors import KNeighborsClassifier
knn1_model = KNeighborsClassifier(n_neighbors=7)
knn1_model.fit(X_train, y_train)
knn1_test_predicted = knn1_model.predict(X_test_final)
knn1_test_predicted_df = pd.DataFrame(knn1_test_predicted,columns=["k-nearest neighbours"])
##dataframe for k nearest neighbours


# In[558]:


#for obtaining perfect k value we will experiment with different k values and check the error vs k value


# In[559]:


error_knn = []
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(X_train,y_train)
    pred_i = knn2.predict(X_test_knn)
    error_knn.append(np.mean(pred_i != y_test_knn))


# In[560]:


plt.figure(figsize=(12,6))
plt.plot(range(1,20),error_knn,color = 'red',linestyle = 'dashed',marker='o',markerfacecolor='blue',markersize = 10)
plt.title('kvalue vs error')
plt.xlabel('kvalue')
plt.ylabel('mean error')


# # error is minimum as k =7

# In[561]:


y_pred = knn1_model.predict(X_test_final)
print("Score is",knn1_model.score(X_train, y_train))


# In[562]:


display_labels_CM = ['dependent_variables','CPL_status']


# In[563]:


#confusion_matrix for knn algorithms
np.set_printoptions(precision=2) #making precision to 2 digits
# Plot non-normalized confusion matrix
titles_options = [("kNN Confusion matrix, without normalization", None),
                  ("KNN Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(knn1_model, X_train, y_train,
                                 display_labels=display_labels_CM,
                                 cmap=plt.cm.BuGn,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()


# In[564]:


y_pred


# In[ ]:





# # using decision tree for classification

# In[565]:


from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_train)
# creating a confusion matrix
cm = confusion_matrix(y_train, dtree_predictions)
dtree_test_predictions = dtree_model.predict(X_test_final)
dtree_test_predictions_df = pd.DataFrame(dtree_test_predictions,columns=["decision_tree"])#data frame for decision tree classifier


# In[566]:


dtree_model.score(X_train,y_train)


# In[567]:


display_labels_CM = ['dependent_variables','CPL_status']


# In[568]:


np.set_printoptions(precision=2) #making precision to 2 digits
# Plot non-normalized confusion matrix
titles_options = [("decision tree Confusion matrix, without normalization", None),
                  ("decision tree Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(dtree_model, X_train, y_train,
                                 display_labels=display_labels_CM,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
plt.show()


# # Naive bayes classifier

# In[569]:


#naive bayes assumes that all the features are independent and have no relation with each other


# In[570]:


from sklearn.naive_bayes import GaussianNB
gnb_model  = GaussianNB().fit(X_train, y_train)
gnb_predictions = gnb_model.predict(X_train)
# accuracy on X_train
accuracy = gnb_model.score(X_train, y_train)
print(accuracy )
# creating a confusion matrix
cm = confusion_matrix(y_train, gnb_predictions)
gaussian_test_predicted = gnb_model.predict(X_test_final)
gaussian_test_predicted_df = pd.DataFrame(gaussian_test_predicted,columns = ["naive bayes classifier"])


# In[571]:


np.set_printoptions(precision=2) #making precision to 2 digits
# Plot non-normalized confusion matrix
titles_options = [("naive bayes Confusion matrix, without normalization", None),
                  ("naive bayes Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(gnb_model, X_train, y_train,
                                 display_labels=display_labels_CM,
                                 cmap=plt.cm.Reds,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()


# # Using Random forests for classification

# In[572]:


rnd_forest_clf = RandomForestClassifier(n_estimators=500,max_leaf_nodes=16)
rnd_forest_class = rnd_forest_clf.fit(X_train,y_train)


# In[573]:


rnd_forest_pred_train = rnd_forest_class.predict(X_train)


# In[574]:


print(rnd_forest_class.score(X_train,y_train)) #score for random forest


# In[575]:


np.set_printoptions(precision=2) #making precision to 2 digits
# Plot non-normalized confusion matrix
titles_options = [("Random Forest Confusion matrix, without normalization", None),
                  ("Random Forest Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(rnd_forest_class, X_train, y_train,
                                 display_labels=display_labels_CM,
                                 cmap=plt.cm.gist_earth_r,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()


# In[576]:


rnf_forest_pred_test = rnd_forest_class.predict(X_test_final)
rnf_forest_pred_test_df = pd.DataFrame(rnf_forest_pred_test,columns=["random forest classifier"])


# In[ ]:





# # Using Random Forests for feature Importance

# In[577]:


print(sorted(zip(map(lambda x:round(x,3),rnd_forest_clf.feature_importances_),onehot_df_with_status.columns),reverse=True))


# In[578]:


sel = RandomForestClassifier(n_estimators=500)
sel.fit(X_train,y_train)


# In[579]:


print(sel.feature_importances_)


# In[580]:


feature_imp = pd.Series(sel.feature_importances_,index=onehot_df.columns)


# In[581]:


feature_imp.plot(kind="barh")
plt.show()


# In[ ]:





# # using ADABoost classifier

# In[582]:


from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=500,algorithm = "SAMME.R",learning_rate=0.5)
ada_clf_model = ada_clf.fit(X_train,y_train)
ada_clf_Xtrain_pred = ada_clf_model.predict(X_train)


# In[583]:


ada_clf_model.score(X_train,y_train)


# In[584]:


ada_clf_test = ada_clf_model.predict(X_test_final)
#lets test accuracy score using accuracy_score
ada_accuracy_score = accuracy_score(ada_clf_Xtrain_pred,y_train)
print(ada_accuracy_score)


# In[585]:


np.set_printoptions(precision=2) #making precision to 2 digits
# Plot non-normalized confusion matrix
titles_options = [("ADABoost Confusion matrix, without normalization", None),
                  ("ADABoost Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(ada_clf_model, X_train, y_train,
                                 display_labels=display_labels_CM,
                                 cmap=plt.cm.gist_stern,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()


# In[586]:


ada_clf_Xtest_pred = ada_clf_model.predict(X_test_final)
ada_clf_Xtest_pred_df = pd.DataFrame(ada_clf_Xtest_pred,columns=["adaboost_predicted"])


# In[587]:


#comparing with support vector classifier
comp_score_svc_adb = accuracy_score(svc_test_predict,ada_clf_Xtest_pred)
print(comp_score_svc_adb)


# # Using XGBoost(Extreme gradient boosting algorithm)

# In[588]:


from xgboost import XGBClassifier
xgb = XGBClassifier()


# In[589]:


xgboost_model = xgb.fit(X_train,y_train)
xgboost_xtrain_predict = xgboost_model.predict(X_train)
xgboost_xtest_predict = xgboost_model.predict(X_test_final)
xgboost_xtest_predict_df = pd.DataFrame(xgboost_xtest_predict,columns=["XGboost_predicted"])


# In[590]:


xgboost_accuracy = accuracy_score(xgboost_xtrain_predict,y_train)
print(xgboost_accuracy * 100)


# In[591]:


np.set_printoptions(precision=2) #making precision to 2 digits
# Plot non-normalized confusion matrix
titles_options = [("XGBoost Confusion matrix, without normalization", None),
                  ("XGBoost Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(xgboost_model, X_train, y_train,
                                 display_labels=display_labels_CM,
                                 cmap=plt.cm.plasma,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()


# # Adding predicted data as columns to test data file

# In[ ]:





# In[592]:


#list of predicted data frames
test_columns = [logreg_test_df ,
svc_test_predict_df,
knn1_test_predicted_df,
gaussian_test_predicted_df,
dtree_test_predictions_df,
ada_clf_Xtest_pred_df,
xgboost_xtest_predict_df]


# In[ ]:





# In[593]:


type(test_columns)


# In[594]:


test_return = pd.concat([testf_data,logreg_test_df,
svc_test_predict_df,
knn1_test_predicted_df,
gaussian_test_predicted_df,
dtree_test_predictions_df,
ada_clf_Xtest_pred_df,
xgboost_xtest_predict_df,rnf_forest_pred_test_df ],axis='columns')


# In[595]:


test_return.columns


# In[596]:


test_submission_columns =  ['logreg', 'svc_predict', 'k-nearest neighbours',
       'naive bayes classifier', 'decision_tree', 'adaboost_predicted',
       'XGboost_predicted', 'random forest classifier']


# In[597]:


for i in test_submission_columns:
    test_return[i] = test_return[i].replace({1:"Y",0:"N"})


# In[598]:


test_return.logreg.replace({1:"Y",0:"N"})


# In[599]:


test_return.columns


# In[600]:


test_return


# # Final test data set

# In[601]:


test_return.to_excel(r'C:\Users\satwik\predicted_test.xlsx')


# In[602]:


type(test_return)


# In[ ]:





# In[ ]:




