# In[1]: Headerimport
from numpy.lib.function_base import average
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data

from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation

from progress.bar import Bar

# In[3]: Atomic mouse actions
from file_names import authentic_match_id

max_tick = 0
min_tick = np.inf
X = []
new_X = []
y = []
counter = 1

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
# bar = Bar('Processing', max=len(authentic_match_id))
for match_id in authentic_match_id:
    df_cursor = pd.read_csv("../data_collector/features/" + match_id + "_cursor_tmp.csv")
    df_unit_order = pd.read_csv("../data_collector/features/" + match_id + "_unit_order.csv")
    df_unit_order.drop_duplicates(inplace=True)
    df_match_info = pd.read_csv("../data_collector/features/" + match_id + "_info.csv")

    dfs_cursor = [rows for _, rows in df_cursor.groupby('Hero')]
    dfs_unit_order = [rows for _, rows in df_unit_order.groupby('Hero')]
    for hero in dfs_unit_order:
        steam_id = df_match_info.loc[df_match_info["Hero"] == hero["Hero"].iloc[0]].iloc[0]["SteamId"]
        hero_name = df_match_info.loc[df_match_info["Hero"] == hero["Hero"].iloc[0]].iloc[0]["Hero"]

        df_hero_cursor = list(filter(lambda x: x.iloc[0]["Hero"] == hero_name, dfs_cursor))[0]
        df_hero_cursor["range"] = pd.cut(df_hero_cursor["Tick"], np.unique(hero["Tick"].values))
        df_hero_cursor.dropna(inplace=True)
        df_hero_cursor["V_X"] = df_hero_cursor["X"].diff() / df_hero_cursor["Tick"].diff()
        df_hero_cursor["V_Y"] = df_hero_cursor["Y"].diff() / df_hero_cursor["Tick"].diff()
        df_hero_cursor["V"] = (df_hero_cursor["V_X"]**2 + df_hero_cursor["V_Y"]**2)**(1/2)
        df_hero_cursor["A"] = df_hero_cursor["V"].diff() / df_hero_cursor["Tick"].diff()
        df_hero_cursor["J"] = df_hero_cursor["A"].diff() / df_hero_cursor["Tick"].diff()
        df_hero_cursor["AoM"] = np.arctan(df_hero_cursor["X"].diff() / df_hero_cursor["Y"].diff())
        df_hero_cursor["AoM"] = df_hero_cursor["AoM"].cumsum()
        df_hero_cursor["Ang_V"] = df_hero_cursor["AoM"].diff() / df_hero_cursor["Tick"].diff()
        df_hero_cursor.fillna({"V_X":0, "V_Y":0, "V":0, "A":0, "J":0, "AoM":0, "Ang_V":0}, inplace=True)
        atomic_order = df_hero_cursor.groupby("range").agg({"Tick": "count", "V_X":["min", "max", "mean", "std"], 
            "V_Y":["min", "max", "mean", "std"],
            "V":["min", "max", "mean", "std"],
            "A":["min", "max", "mean", "std"],
            "J":["min", "max", "mean", "std"],
            "AoM":["min", "max", "mean", "std"],
            "Ang_V":["min", "max", "mean", "std"],
            }).fillna(0).replace([np.inf, -np.inf], 0)
        atomic_order.drop(atomic_order[atomic_order["Tick"]["count"] < 20].index, inplace = True)
        atomic_order_arr = atomic_order.to_numpy().flatten()
        # max_tick = atomic_order_arr.size if atomic_order_arr.size > max_tick else max_tick
        if atomic_order_arr.size < 2000:
            continue
        min_tick = atomic_order_arr.size if atomic_order_arr.size < min_tick else min_tick
        
        if steam_id == 76561198134243802:
            if hero_name == "CDOTA_Unit_Hero_Puck":
                y.append(1)
            else: continue
        else:
            y.append(0)
        
        new_X.append(atomic_order_arr)

            # if row["Action"] == "M":
        # move_order.append(new_row, ignore_index=True)
            # elif row["Action"] == "A":
            #     attack_order.append(new_row, ignore_index=True)
            # elif row["Action"] == "S":
            #     spell_order.append(new_row, ignore_index=True)
    # bar.next()
    print("batch " + str(counter) + "/" + str(len(authentic_match_id)))
    counter += 1
new_X_padded  = list(map(lambda x: np.resize(x, min_tick), new_X))
X_train, X_test, y_train, y_test = train_test_split(new_X_padded, y, test_size=0.20, random_state=42)
# bar.finish()

# In[2]: Read data and split train and test data
# from file_names import authentic_match_id

# max_tick = 0
# X = []
# y = []
# new_X = []
# counter = 0 
# for match_id in authentic_match_id:
#     df_cursor = pd.read_csv("../data_collector/features/" + match_id + "_cursor.csv")
#     df_match_info = pd.read_csv("../data_collector/features/" + match_id + "_info.csv")

#     dfs = [rows for _, rows in df_cursor.groupby('Hero')]
#     for cursor_info in dfs:
#         steam_id = df_match_info.loc[df_match_info["Hero"] == cursor_info["Hero"].iloc[0]].iloc[0]["SteamId"]
#         hero_name = df_match_info.loc[df_match_info["Hero"] == cursor_info["Hero"].iloc[0]].iloc[0]["Hero"]
#         max_tick = cursor_info.shape[0] if cursor_info.shape[0] > max_tick else max_tick
#         if steam_id == 76561198134243802:
#             if hero_name == "CDOTA_Unit_Hero_Puck":
#                 y.append(1)
#                 X.append(cursor_info.drop("Hero", axis=1))
#         else:
#             X.append(cursor_info.drop("Hero", axis=1))
#             y.append(0)
        
# for cursor_info in X:
#     cursor_info["V_X"] = cursor_info["X"].diff() / cursor_info["Tick"].diff()
#     cursor_info["V_Y"] = cursor_info["Y"].diff() / cursor_info["Tick"].diff()
#     cursor_info["V"] = (cursor_info["V_X"]**2 + cursor_info["V_Y"]**2)**(1/2)
#     cursor_info["A"] = cursor_info["V"].diff() / cursor_info["Tick"].diff()
#     cursor_info["J"] = cursor_info["A"].diff() / cursor_info["Tick"].diff()
#     cursor_info["AoM"] = np.arctan(cursor_info["X"].diff() / cursor_info["Y"].diff())
#     cursor_info["AoM"] = cursor_info["AoM"].cumsum()
#     cursor_info["Ang_V"] = cursor_info["AoM"].diff() / cursor_info["Tick"].diff()
#     cursor_info.fillna(0, inplace=True)
#     cursor_info.drop("Tick", axis=1, inplace=True)
#     cursor_info.drop("X", axis=1, inplace=True)
#     cursor_info.drop("Y", axis=1, inplace=True)
#     cursor_info.replace([np.inf, -np.inf], 0, inplace=True)
#     if cursor_info.shape[0] < max_tick:
#         pddddd = pd.DataFrame({
#                                 # "Tick": np.zeros(max_tick-cursor_info.shape[0]),
#                                 # "X": np.zeros(max_tick-cursor_info.shape[0]), 
#                                 # "Y": np.zeros(max_tick-cursor_info.shape[0]), 
#                                 "V_X": np.zeros(max_tick-cursor_info.shape[0]), 
#                                 "V_Y": np.zeros(max_tick-cursor_info.shape[0]), 
#                                 "V": np.zeros(max_tick-cursor_info.shape[0]), 
#                                 "A": np.zeros(max_tick-cursor_info.shape[0]), 
#                                 "J": np.zeros(max_tick-cursor_info.shape[0]), 
#                                 "AoM": np.zeros(max_tick-cursor_info.shape[0]), 
#                                 "Ang_V": np.zeros(max_tick-cursor_info.shape[0])})

#         cursor_info = cursor_info.append(pddddd, ignore_index = True)
#     new_X.append(cursor_info.to_numpy().flatten())
    
# X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.20, random_state=42)

# In[3]: Calculate eer rate
from sklearn.metrics import make_scorer, roc_curve #for eer
from scipy.optimize import brentq #for eer
from scipy.interpolate import interp1d #for eer
def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


# In[4]: Logistic Regression
print("fine")
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=42).fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('The accuracy of the Logistic Regression is ', round(accuracy_score(prediction_rm, y_test)*100,2))
print('The precision of the Logistic Regression is ', round(precision_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The recall of the Logistic Regression is ', round(recall_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The f1_score of the Logistic Regression is ', round(f1_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
# print('The EER value of the Logistic Regression is ', round(calculate_eer(prediction_rm, y_test)*100,2))
print("fine")

kfold = KFold(n_splits=5) # k=5, split the data into 5 equal parts
result_rm=cross_val_score(clf, new_X_padded, y, cv=5,scoring='accuracy')
print('----------------------The cross validated accuracy score for Logistic Regression is:',round(result_rm.mean()*100,2))
result_rm=cross_val_score(clf, new_X_padded, y, cv=5,scoring='precision')
print('----------------------The cross validated precision score for Logistic Regression is:',round(result_rm.mean()*100,2))
result_rm=cross_val_score(clf, new_X_padded, y, cv=5,scoring='recall')
print('----------------------The cross validated recall score for Logistic Regression is:',round(result_rm.mean()*100,2))


# In[5]: Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('The accuracy of the Decision Tree is ', round(accuracy_score(prediction_rm, y_test)*100,2))
print('The precision of the Decision Tree is ', round(precision_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The recall of the Decision Tree is ', round(recall_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The f1_score of the Decision Tree is ', round(f1_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
# print('The EER value of the Decision Tree is ', round(calculate_eer(prediction_rm, y_test)*100,2))

kfold = KFold(n_splits=5) # k=5, split the data into 5 equal parts
result_rm=cross_val_score(clf, new_X_padded, y, cv=5,scoring='accuracy')
print('----------------------The cross validated score for Decision Tree is:',round(result_rm.mean()*100,2))


# In[6]: Random Forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('The accuracy of the Random Forest is ', round(accuracy_score(prediction_rm, y_test)*100,2))
print('The precision of the Random Forest is ', round(precision_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The recall of the Random Forest is ', round(recall_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The f1_score of the Random Forest is ', round(f1_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
# print('The EER value of the Random Forest is ', round(calculate_eer(prediction_rm, y_test)*100,2))

kfold = KFold(n_splits=5) # k=5, split the data into 5 equal parts
result_rm=cross_val_score(clf, new_X_padded, y, cv=5,scoring='accuracy')
print('----------------------The cross validated score for Random Forest is:',round(result_rm.mean()*100,2))


# %%
