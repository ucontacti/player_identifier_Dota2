# In[1]: Headerimport
from operator import mul
from numpy.lib.function_base import append, average
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data

from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_validate #score evaluation


# In[3]: Atomic mouse actions
from file_names import authentic_match_id

new_X = []
counter = 1
new_X_mov = []
new_X_att = []
new_X_spl = []
y = []

for match_id in authentic_match_id:
    df_action = pd.read_csv("../data_collector/features/" + match_id + "-1278641240.dem.bz2-mouseaction.csv")
    if df_action.empty:
        print(match_id)
        continue
    df_action.drop("actionType", axis=1, inplace=True)
    steam_id = df_action["steamid"].iloc[0]
    df_action.drop("steamid", axis=1, inplace=True)
    df_action.fillna(0, inplace=True)
    df_action.replace([np.inf, -np.inf], 0, inplace=True)
    new_X.append(df_action.to_numpy().flatten())
    if steam_id == 76561198162034645:
            y.append(1)
    else:
        y.append(0)
    # print("batch " + str(counter) + "/" + str(len(authentic_match_id)))
    counter += 1
med_tick = 240000
new_X_padded  = list(map(lambda x: np.resize(x, med_tick) if np.size(x) >= med_tick else np.pad(x, (0, med_tick - x.size), 'constant'), new_X))
X_train, X_test, y_train, y_test = train_test_split(new_X_padded, y, test_size=0.25, random_state=42)


# In[4]: Logistic Regression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=42, max_iter=200).fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('The accuracy of the Logistic Regression is ', round(accuracy_score(prediction_rm, y_test)*100,2))
print('The precision of the Logistic Regression is ', round(precision_score(prediction_rm, y_test, pos_label=1)*100,2))
print('The recall of the Logistic Regression is ', round(recall_score(prediction_rm, y_test, pos_label=1)*100,2))
print('The f1_score of the Logistic Regression is ', round(f1_score(prediction_rm, y_test, pos_label=1)*100,2))
# print('The micro precision of the Logistic Regression is ', round(precision_score(prediction_rm, y_test, pos_label=1, average='micro')*100,2))
# print('The micro recall of the Logistic Regression is ', round(recall_score(prediction_rm, y_test, pos_label=1, average='micro')*100,2))
# print('The micro f1_score of the Logistic Regression is ', round(f1_score(prediction_rm, y_test, pos_label=1, average='micro')*100,2))
# print('The EER value of the Logistic Regression is ', round(calculate_eer(prediction_rm, y_test)*100,2))

result_rm=cross_validate(clf, new_X_padded, y, cv=5,scoring=['precision', 'recall', 'accuracy', 'f1'])
# result_rm=cross_val_score(clf, new_X_padded, y, cv=5,scoring='accuracy')
print('----------------------The cross validated precision score for Logistic Regression is:',round(result_rm["test_precision"].mean()*100,2))
print('----------------------The cross validated recall score for Logistic Regression is:',round(result_rm["test_recall"].mean()*100,2))
print('----------------------The cross validated f1 score for Logistic Regression is:',round(result_rm["test_f1"].mean()*100,2))
# result_rm=cross_val_score(clf, new_X_padded, y, cv=5,scoring='precision')
# print('----------------------The cross validated precision score for Logistic Regression is:',round(result_rm.mean()*100,2))
# result_rm=cross_val_score(clf, new_X_padded, y, cv=5,scoring='recall')
# print('----------------------The cross validated recall score for Logistic Regression is:',round(result_rm.mean()*100,2))

# In[4]: Logistic Regression CV
from sklearn.linear_model import LogisticRegressionCV

clf = LogisticRegressionCV(cv=5, random_state=42, max_iter=200).fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('The accuracy of the Logistic Regression is ', round(accuracy_score(prediction_rm, y_test)*100,2))
print('The precision of the Logistic Regression is ', round(precision_score(prediction_rm, y_test, pos_label=1)*100,2))
print('The recall of the Logistic Regression is ', round(recall_score(prediction_rm, y_test, pos_label=1)*100,2))
print('The f1_score of the Logistic Regression is ', round(f1_score(prediction_rm, y_test, pos_label=1)*100,2))
# print('The macro precision of the Logistic Regression is ', round(precision_score(prediction_rm, y_test, pos_label=1, average='macro')*100,2))
# print('The macro recall of the Logistic Regression is ', round(recall_score(prediction_rm, y_test, pos_label=1, average='macro')*100,2))
# print('The macro f1_score of the Logistic Regression is ', round(f1_score(prediction_rm, y_test, pos_label=1, average='macro')*100,2))
# print('The EER value of the Logistic Regression is ', round(calculate_eer(prediction_rm, y_test)*100,2))

# kfold = KFold(n_splits=5) # k=5, split the data into 5 equal parts
# # result_rm=cross_val_score(clf, new_X_padded, y, cv=5,scoring='accuracy')
# # print('----------------------The cross validated accuracy score for Logistic Regression is:',round(result_rm.mean()*100,2))
# result_rm=cross_val_score(clf, new_X_padded, y, cv=5,scoring='precision')
# print('----------------------The cross validated precision score for Logistic Regression is:',round(result_rm.mean()*100,2))
# result_rm=cross_val_score(clf, new_X_padded, y, cv=5,scoring='recall')
# print('----------------------The cross validated recall score for Logistic Regression is:',round(result_rm.mean()*100,2))

# In[5]: Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('The accuracy of the Decision Tree is ', round(accuracy_score(prediction_rm, y_test)*100,2))
print('The precision of the Decision Tree is ', round(precision_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The recall of the Decision Tree is ', round(recall_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The f1_score of the Decision Tree is ', round(f1_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
# print('The EER value of the Decision Tree is ', round(calculate_eer(prediction_rm, y_test)*100,2))
# print('The macro precision of the Decision Tree is ', round(precision_score(prediction_rm, y_test, pos_label=1, average='macro')*100,2))
# print('The macro recall of the Decision Tree is ', round(recall_score(prediction_rm, y_test, pos_label=1, average='macro')*100,2))
# print('The macro f1_score of the Decision Tree is ', round(f1_score(prediction_rm, y_test, pos_label=1, average='macro')*100,2))


kfold = KFold(n_splits=5) # k=5, split the data into 5 equal parts
# result_rm=cross_val_score(clf, new_X_padded, y, cv=5,scoring='accuracy')
# print('----------------------The cross validated accuracy score for Decision Tree is:',round(result_rm.mean()*100,2))
result_rm=cross_val_score(clf, new_X_padded, y, cv=5,scoring='precision')
print('----------------------The cross validated precision score for Decision Tree is:',round(result_rm.mean()*100,2))
# result_rm=cross_val_score(clf, new_X_padded, y, cv=5,scoring='recall')
# print('----------------------The cross validated recall score for Decision Tree is:',round(result_rm.mean()*100,2))


# In[6]: Random Forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('The accuracy of the Random Forest is ', round(accuracy_score(prediction_rm, y_test)*100,2))
print('The precision of the Random Forest is ', round(precision_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The recall of the Random Forest is ', round(recall_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The f1_score of the Random Forest is ', round(f1_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
# print('The EER value of the Random Forest is ', round(calculate_eer(prediction_rm, y_test)*100,2))


# print('The macro precision of the Random Forset is ', round(precision_score(prediction_rm, y_test, pos_label=1, average='macro')*100,2))
# print('The macro recall of the Random Forset is ', round(recall_score(prediction_rm, y_test, pos_label=1, average='macro')*100,2))
# print('The macro f1_score of the Random Forset is ', round(f1_score(prediction_rm, y_test, pos_label=1, average='macro')*100,2))
kfold = KFold(n_splits=5) # k=5, split the data into 5 equal parts
# result_rm=cross_val_score(clf, new_X_padded, y, cv=5,scoring='accuracy')
# print('----------------------The cross validated score for Random Forest is:',round(result_rm.mean()*100,2))
result_rm=cross_val_score(clf, new_X_padded, y, cv=5,scoring='precision')
print('----------------------The cross validated precision score for Random Forest is:',round(result_rm.mean()*100,2))


# %%
