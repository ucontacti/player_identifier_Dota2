# In[1]: Headerimport
from numpy.lib.function_base import average
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data

from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation


# In[3]: Atomic mouse actions
from file_names import authentic_match_id


max_tick = 0
X = []
counter = 0 

move_order = pd.DataFrame(columns=[
    "Hero", "SteamId",
    "V_X_min", "V_X_max", "V_X_mean", "V_X_std", 
    "V_Y_min", "V_Y_max", "V_Y_mean", "V_Y_std", 
    "V_min", "V_max", "V_mean", "V_std", 
    "A_min", "A_max", "A_mean", "A_std", 
    "J_min", "J_max", "J_mean", "J_std", 
    "AoM_min", "AoM_max", "AoM_mean", "AoM_std", 
    "Ang_V_min", "Ang_V_max", "Ang_V_mean", "Ang_V_std", 
    "tick_delta", "d" 
    ])
attack_order = pd.DataFrame(columns=[
    "Hero", "SteamId",
    "V_X_min", "V_X_max", "V_X_mean", "V_X_std", 
    "V_Y_min", "V_Y_max", "V_Y_mean", "V_Y_std", 
    "V_min", "V_max", "V_mean", "V_std", 
    "A_min", "A_max", "A_mean", "A_std", 
    "J_min", "J_max", "J_mean", "J_std", 
    "AoM_min", "AoM_max", "AoM_mean", "AoM_std", 
    "Ang_V_min", "Ang_V_max", "Ang_V_mean", "Ang_V_std", 
    "tick_delta", "d" 
    ])
spell_order = pd.DataFrame(columns=[
    "Hero", "SteamId",
    "V_X_min", "V_X_max", "V_X_mean", "V_X_std", 
    "V_Y_min", "V_Y_max", "V_Y_mean", "V_Y_std", 
    "V_min", "V_max", "V_mean", "V_std", 
    "A_min", "A_max", "A_mean", "A_std", 
    "J_min", "J_max", "J_mean", "J_std", 
    "AoM_min", "AoM_max", "AoM_mean", "AoM_std", 
    "Ang_V_min", "Ang_V_max", "Ang_V_mean", "Ang_V_std", 
    "tick_delta", "d" 
    ])

for match_id in authentic_match_id:
    df_cursor = pd.read_csv("../data_collector/features/" + match_id + "_cursor_tmp.csv")
    df_unit_order = pd.read_csv("../data_collector/features/" + match_id + "_unit_order.csv")
    df_unit_order.drop_duplicates(inplace=True)
    df_match_info = pd.read_csv("../data_collector/features/" + match_id + "_info.csv")

    # dfs_cursor = [rows for _, rows in df_cursor.groupby('Hero')]
    dfs_unit_order = [rows for _, rows in df_unit_order.groupby('Hero')]
    for hero in dfs_unit_order:
        beginning = True
        steam_id = df_match_info.loc[df_match_info["Hero"] == hero["Hero"].iloc[0]].iloc[0]["SteamId"]
        hero_name = df_match_info.loc[df_match_info["Hero"] == hero["Hero"].iloc[0]].iloc[0]["Hero"]
        for index, row in hero.iterrows():

            if beginning == True:
                prev_row = row
                beginning = False
                continue
            df_ticks = df_cursor[(df_cursor["Tick"] > prev_row["Tick"]) & (df_cursor["Tick"] <= row["Tick"]) & (row["Hero"] == df_cursor["Hero"])]
            
            prev_row = row
            if df_ticks.empty:
                continue
            df_ticks["V_X"] = df_ticks["X"].diff() / df_ticks["Tick"].diff()
            df_ticks["V_Y"] = df_ticks["Y"].diff() / df_ticks["Tick"].diff()
            df_ticks["V"] = (df_ticks["V_X"]**2 + df_ticks["V_Y"]**2)**(1/2)
            df_ticks["A"] = df_ticks["V"].diff() / df_ticks["Tick"].diff()
            df_ticks["J"] = df_ticks["A"].diff() / df_ticks["Tick"].diff()
            df_ticks["AoM"] = np.arctan(df_ticks["X"].diff() / df_ticks["Y"].diff())
            df_ticks["AoM"] = df_ticks["AoM"].cumsum()
            df_ticks["Ang_V"] = df_ticks["AoM"].diff() / df_ticks["Tick"].diff()
            df_ticks.fillna(0, inplace=True)
            p_init = pd.Series([df_ticks.iloc[0]["X"], df_ticks.iloc[0]["Y"]])
            p_fin = pd.Series([df_ticks.iloc[-1]["X"], df_ticks.iloc[-1]["Y"]])
            new_row = {"Hero": hero_name, "SteamId": steam_id,
                    "V_X_min":df_ticks["V_X"].min(), "V_X_max":df_ticks["V_X"].max(), "V_X_mean":df_ticks["V_X"].mean(), "V_X_std":df_ticks["V_X"].std(),
                    "V_Y_min":df_ticks["V_Y"].min(), "V_Y_max":df_ticks["V_Y"].max(), "V_Y_mean":df_ticks["V_Y"].mean(), "V_Y_std":df_ticks["V_Y"].std(),
                    "V_min":df_ticks["V"].min(), "V_max":df_ticks["V"].max(), "V_mean":df_ticks["V"].mean(), "V_std":df_ticks["V"].std(),
                    "A_min":df_ticks["A"].min(), "A_max":df_ticks["A"].max(), "A_mean":df_ticks["A"].mean(), "A_std":df_ticks["A"].std(),
                    "J_min":df_ticks["J"].min(), "J_max":df_ticks["J"].max(), "J_mean":df_ticks["J"].mean(), "J_std":df_ticks["J"].std(),
                    "AoM_min":df_ticks["AoM"].min(), "AoM_max":df_ticks["AoM"].max(), "AoM_mean":df_ticks["AoM"].mean(), "AoM_std":df_ticks["AoM"].std(),
                    "Ang_V_min":df_ticks["Ang_V"].min(), "Ang_V_max":df_ticks["Ang_V"].max(), "Ang_V_mean":df_ticks["Ang_V"].mean(), "Ang_V_std":df_ticks["Ang_V"].std(),
                    "tick_delta": df_ticks.shape[0], "d": np.linalg.norm(p_init - p_fin)
                }
            if row["Action"] == "M":
                move_order.append(new_row, ignore_index=True)
            elif row["Action"] == "A":
                attack_order.append(new_row, ignore_index=True)
            elif row["Action"] == "S":
                spell_order.append(new_row, ignore_index=True)

    
print("Done")
# In[2]: Read data and split train and test data
from file_names import authentic_match_id

max_tick = 0
X = []
y = []
new_X = []
counter = 0 
for match_id in authentic_match_id:
    df_cursor = pd.read_csv("../data_collector/features/" + match_id + "_cursor.csv")
    df_match_info = pd.read_csv("../data_collector/features/" + match_id + "_info.csv")

    dfs = [rows for _, rows in df_cursor.groupby('Hero')]
    for cursor_info in dfs:
        steam_id = df_match_info.loc[df_match_info["Hero"] == cursor_info["Hero"].iloc[0]].iloc[0]["SteamId"]
        hero_name = df_match_info.loc[df_match_info["Hero"] == cursor_info["Hero"].iloc[0]].iloc[0]["Hero"]
        max_tick = cursor_info.shape[0] if cursor_info.shape[0] > max_tick else max_tick
        if steam_id == 76561198134243802:
            if hero_name == "CDOTA_Unit_Hero_Puck":
                y.append(1)
                X.append(cursor_info.drop("Hero", axis=1))
        else:
            X.append(cursor_info.drop("Hero", axis=1))
            y.append(0)
        
for cursor_info in X:
    cursor_info["V_X"] = cursor_info["X"].diff() / cursor_info["Tick"].diff()
    cursor_info["V_Y"] = cursor_info["Y"].diff() / cursor_info["Tick"].diff()
    cursor_info["V"] = (cursor_info["V_X"]**2 + cursor_info["V_Y"]**2)**(1/2)
    cursor_info["A"] = cursor_info["V"].diff() / cursor_info["Tick"].diff()
    cursor_info["J"] = cursor_info["A"].diff() / cursor_info["Tick"].diff()
    cursor_info["AoM"] = np.arctan(cursor_info["X"].diff() / cursor_info["Y"].diff())
    cursor_info["AoM"] = cursor_info["AoM"].cumsum()
    cursor_info["Ang_V"] = cursor_info["AoM"].diff() / cursor_info["Tick"].diff()
    cursor_info.fillna(0, inplace=True)
    cursor_info.drop("Tick", axis=1, inplace=True)
    cursor_info.drop("X", axis=1, inplace=True)
    cursor_info.drop("Y", axis=1, inplace=True)
    cursor_info.replace([np.inf, -np.inf], 0, inplace=True)
    if cursor_info.shape[0] < max_tick:
        pddddd = pd.DataFrame({
                                # "Tick": np.zeros(max_tick-cursor_info.shape[0]),
                                # "X": np.zeros(max_tick-cursor_info.shape[0]), 
                                # "Y": np.zeros(max_tick-cursor_info.shape[0]), 
                                "V_X": np.zeros(max_tick-cursor_info.shape[0]), 
                                "V_Y": np.zeros(max_tick-cursor_info.shape[0]), 
                                "V": np.zeros(max_tick-cursor_info.shape[0]), 
                                "A": np.zeros(max_tick-cursor_info.shape[0]), 
                                "J": np.zeros(max_tick-cursor_info.shape[0]), 
                                "AoM": np.zeros(max_tick-cursor_info.shape[0]), 
                                "Ang_V": np.zeros(max_tick-cursor_info.shape[0])})

        cursor_info = cursor_info.append(pddddd, ignore_index = True)
    new_X.append(cursor_info.to_numpy().flatten())
    
X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.20, random_state=15)

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

clf = LogisticRegression(random_state=15).fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('The accuracy of the Logistic Regression is ', round(accuracy_score(prediction_rm, y_test)*100,2))
print('The precision of the Logistic Regression is ', round(precision_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The recall of the Logistic Regression is ', round(recall_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The f1_score of the Logistic Regression is ', round(f1_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
# print('The EER value of the Logistic Regression is ', round(calculate_eer(prediction_rm, y_test)*100,2))

# kfold = KFold(n_splits=5) # k=5, split the data into 5 equal parts
# result_rm=cross_val_score(clf, new_X, y, cv=5,scoring='accuracy')
# print('The cross validated score for Logistic Regression is:',round(result_rm.mean()*100,2))


# In[5]: Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=15).fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('The accuracy of the Decision Tree is ', round(accuracy_score(prediction_rm, y_test)*100,2))
print('The precision of the Decision Tree is ', round(precision_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The recall of the Decision Tree is ', round(recall_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The f1_score of the Decision Tree is ', round(f1_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
# print('The EER value of the Decision Tree is ', round(calculate_eer(prediction_rm, y_test)*100,2))

# kfold = KFold(n_splits=5) # k=5, split the data into 5 equal parts
# result_rm=cross_val_score(clf, new_X, y, cv=5,scoring='accuracy')
# print('The cross validated score for Decision Tree is:',round(result_rm.mean()*100,2))


# In[6]: Random Forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=15).fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('The accuracy of the Random Forest is ', round(accuracy_score(prediction_rm, y_test)*100,2))
print('The precision of the Random Forest is ', round(precision_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The recall of the Random Forest is ', round(recall_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The f1_score of the Random Forest is ', round(f1_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
# print('The EER value of the Random Forest is ', round(calculate_eer(prediction_rm, y_test)*100,2))

# kfold = KFold(n_splits=5) # k=5, split the data into 5 equal parts
# result_rm=cross_val_score(clf, new_X, y, cv=5,scoring='accuracy')
# print('The cross validated score for Random Forest is:',round(result_rm.mean()*100,2))


# %%
