# In[1]: Headerimport
from numpy.lib.function_base import average
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data


# In[2]: Read data and split train and test data

authentic_match_id = ["5910973712", "5911045449", "5912945803", "5913000523", "5915392078"]
max_tick = 0
X = []
y = []
new_X = []
for match_id in authentic_match_id:
# match_id = authentic_match_id[3]
    df_cursor = pd.read_csv("../data_collector/features/" + match_id + "_cursor.csv")
    df_match_info = pd.read_csv("../data_collector/features/" + match_id + "_info.csv")

    dfs = [rows for _, rows in df_cursor.groupby('Hero')]
    for cursor_info in dfs:
        steam_id = df_match_info.loc[df_match_info["Hero"] == cursor_info["Hero"].iloc[0]].iloc[0]["SteamId"]
        max_tick = cursor_info.shape[0] if cursor_info.shape[0] > max_tick else max_tick
        X.append(cursor_info.drop("Hero", axis=1).drop("Tick", axis=1))
        if steam_id == 76561198134243802:
            y.append(1)
        else:
            y.append(0)
for cursor_info in X:
    if cursor_info.shape[0] < max_tick:
        pddddd = pd.DataFrame({
                                "X": np.zeros(max_tick-cursor_info.shape[0]), 
                                "Y": np.zeros(max_tick-cursor_info.shape[0])})

        cursor_info = cursor_info.append(pddddd, ignore_index = True)
    new_X.append(cursor_info.to_numpy().flatten())
    
X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.20, random_state=24)

# In[3]: Calculate eer rate
from sklearn.metrics import make_scorer, roc_curve #for eer
from scipy.optimize import brentq #for eer
from scipy.interpolate import interp1d #for eer
def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer



# In[4]: Logistic Regression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=24).fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('The accuracy of the Logistic Regression is ', round(accuracy_score(prediction_rm, y_test)*100,2))
print('The precision of the Logistic Regression is ', round(precision_score(prediction_rm, y_test, average="macro")*100,2))
print('The recall of the Logistic Regression is ', round(recall_score(prediction_rm, y_test, average="macro")*100,2))
print('The EER value of the Logistic Regression is ', round(calculate_eer(prediction_rm, y_test)*100,2))

# In[4]: Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=24).fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('The accuracy of the Decision Tree is ', round(accuracy_score(prediction_rm, y_test)*100,2))
print('The precision of the Decision Tree is ', round(precision_score(prediction_rm, y_test, average="macro")*100,2))
print('The recall of the Decision Tree is ', round(recall_score(prediction_rm, y_test, average="macro")*100,2))
print('The EER value of the Decision Tree is ', round(calculate_eer(prediction_rm, y_test)*100,2))

# %%
