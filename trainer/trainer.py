# In[1]: Headerimport
from operator import mul
from numpy.lib.function_base import append, average
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data

from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation

from progress.bar import Bar

# In[3]: Atomic mouse actions
from file_names import authentic_match_id

new_X = []
counter = 1
new_X_mov = []
new_X_att = []
new_X_spl = []


# bar = Bar('Processing', max=len(authentic_match_id))
for match_id in authentic_match_id:
    df_cursor = pd.read_csv("../data_collector/features/" + match_id + "_cursor_tmp.csv")
    df_unit_order = pd.read_csv("../data_collector/features/" + match_id + "_unit_order_v2.csv")
    df_unit_order.drop_duplicates(inplace=True)
    df_unit_order.replace({"Action": {'M': 1, 'A': 2, 'S': 3},}, inplace=True)
    df_match_info = pd.read_csv("../data_collector/features/" + match_id + "_info.csv")

    dfs_cursor = [rows for _, rows in df_cursor.groupby('Hero')]
    dfs_unit_order = [rows for _, rows in df_unit_order.groupby('Hero')]
    for hero in dfs_unit_order:
        steam_id = df_match_info.loc[df_match_info["Hero"] == hero["Hero"].iloc[0]].iloc[0]["SteamId"]
        hero_name = df_match_info.loc[df_match_info["Hero"] == hero["Hero"].iloc[0]].iloc[0]["Hero"]
        hero = hero.groupby(hero['Tick']).aggregate({'Action': 'max'}).reset_index()
        df_hero_cursor = list(filter(lambda x: x.iloc[0]["Hero"] == hero_name, dfs_cursor))[0]
        df_hero_cursor["Action"] = pd.cut(df_hero_cursor["Tick"], bins=hero["Tick"].values, labels=hero["Action"].iloc[1:].values, ordered=False)
        df_hero_cursor["range"] = pd.cut(df_hero_cursor["Tick"], hero["Tick"].values)
        df_hero_cursor.dropna(inplace=True)
        df_hero_cursor["V_X"] = df_hero_cursor["X"].diff() / df_hero_cursor["Tick"].diff()
        df_hero_cursor["V_Y"] = df_hero_cursor["Y"].diff() / df_hero_cursor["Tick"].diff()
        df_hero_cursor["V"] = np.sqrt(df_hero_cursor["V_X"]**2 + df_hero_cursor["V_Y"]**2)
        df_hero_cursor["S"] = np.sqrt((df_hero_cursor["X"].diff())**2 + (df_hero_cursor["Y"].diff())**2)
        df_hero_cursor.fillna({"S":0}, inplace=True)
        df_hero_cursor["S"] = np.cumsum(df_hero_cursor["S"])
        df_hero_cursor["A"] = df_hero_cursor["V"].diff() / df_hero_cursor["Tick"].diff()
        df_hero_cursor["J"] = df_hero_cursor["A"].diff() / df_hero_cursor["Tick"].diff()
        df_hero_cursor["AoM"] = np.arctan(df_hero_cursor["Y"].diff() / df_hero_cursor["X"].diff())
        df_hero_cursor.fillna({"AoM":0}, inplace=True)
        df_hero_cursor["AoM"] = np.cumsum(df_hero_cursor["AoM"])
        df_hero_cursor["Ang_V"] = df_hero_cursor["AoM"].diff() / df_hero_cursor["Tick"].diff()
        df_hero_cursor["Cur"] = df_hero_cursor["AoM"].diff() / df_hero_cursor["S"].diff()
        df_hero_cursor["Y_diff"] = df_hero_cursor["Y"].diff()
        df_hero_cursor["X_diff"] = df_hero_cursor["X"].diff()
        # df_hero_cursor.drop("S", axis=0, inplace=True)
        # df_hero_cursor["Cur_cr"] = df_hero_cursor["Cur"].diff() / df_hero_cursor["S"].diff()
        df_hero_cursor.fillna({"V_X":0, "V_Y":0, "V":0, "A":0, "J":0, "AoM":0, "Ang_V":0, "Cur":0, "Y_diff":0, "X_diff":0}, inplace=True)
        atomic_order = df_hero_cursor.groupby("range").agg({
            "Tick": "count", 
            "Action": "first",
            "X_diff": "sum",
            "Y_diff": "sum",
            "V_X":["min", "max", "mean", "std"], 
            "V_Y":["min", "max", "mean", "std"],
            "V":["min", "max", "mean", "std"],
            "A":["min", "max", "mean", "std"],
            "J":["min", "max", "mean", "std"],
            "AoM":["min", "max", "mean", "std"],
            "Ang_V":["min", "max", "mean", "std"],
            "Cur":["min", "max", "mean", "std"]
            # "Cur_cr":["min", "max", "mean", "std"],
            }).fillna(0).replace([np.inf, -np.inf], 0)
        atomic_order["distance"] = np.sqrt(atomic_order["X_diff"]["sum"]**2 + atomic_order["Y_diff"]["sum"]**2)
        atomic_order.drop("X_diff", axis=1 ,inplace=True)
        atomic_order.drop("Y_diff", axis=1 ,inplace=True)
        atomic_order.drop(atomic_order[atomic_order["Tick"]["count"] < 20].index, inplace = True)
        atomic_order_move = atomic_order.drop(atomic_order[atomic_order["Action"]["first"] != 1].index)
        atomic_order_attack = atomic_order.drop(atomic_order[atomic_order["Action"]["first"] != 2].index)
        atomic_order_spell = atomic_order.drop(atomic_order[atomic_order["Action"]["first"] != 3].index)
        atomic_order_move_arr = [[steam_id, hero_name], atomic_order_move.to_numpy().flatten()]
        atomic_order_attack_arr = [[steam_id, hero_name], atomic_order_attack.to_numpy().flatten()]
        atomic_order_spell_arr = [[steam_id, hero_name], atomic_order_spell.to_numpy().flatten()]
        atomic_order_arr = [[steam_id, hero_name], atomic_order.to_numpy().flatten()]
        new_X_mov.append(atomic_order_move_arr)
        new_X_att.append(atomic_order_attack_arr)
        new_X_spl.append(atomic_order_spell_arr)
        new_X.append(atomic_order_arr)


    # bar.next()
    print("batch " + str(counter) + "/" + str(len(authentic_match_id)))
    counter += 1
np.save('atomic_move_v3_8.npy', new_X_mov, allow_pickle=True)
np.save('atomic_attack_v3_8.npy', new_X_att, allow_pickle=True)
np.save('atomic_spell_v3_8.npy', new_X_spl, allow_pickle=True)
np.save('atomic_v3_8.npy', new_X, allow_pickle=True)

# bar.finish()

# In[3]: Trim data to our need
X = np.concatenate((np.load('atomic_v3_1.npy', allow_pickle=True), 
                    np.load('atomic_v3_2.npy', allow_pickle=True), 
                    np.load('atomic_v3_3.npy', allow_pickle=True), 
                    np.load('atomic_v3_4.npy', allow_pickle=True), 
                    np.load('atomic_v3_5.npy', allow_pickle=True), 
                    np.load('atomic_v3_6.npy', allow_pickle=True), 
                    # np.load('atomic_v3_7.npy', allow_pickle=True), 
                    np.load('atomic_v3_8.npy', allow_pickle=True), 
                    np.load('atomic_v3_9.npy', allow_pickle=True),
                    np.load('atomic_v3_10.npy', allow_pickle=True)))

new_X = []
max_tick = 0
min_tick = np.inf
y = []


for inst in X:
    atomic_inst = inst[1]
    atomic_inst = np.delete(atomic_inst, np.arange(1, atomic_inst.size, 35))
    steam_id = inst[0][0]
    hero_name = inst[0][1]
    if atomic_inst.size < 2000 or atomic_inst.size > 80000:
        continue
    # max_tick = atomic_inst.size if atomic_inst.size > max_tick else max_tick
    # min_tick = atomic_inst.size if atomic_inst.size < min_tick else min_tick
    if steam_id == 76561198162034645:
        # if hero_name == "CDOTA_Unit_Hero_Obsidian_Destroyer":
            y.append(1)
        # else: continue
    # elif steam_id == 76561198173337033:
    else:
        y.append(0)
    # else: continue
    new_X.append(atomic_inst)
med_tick = 20000
# new_X_padded  = list(map(lambda x: np.resize(x, min_tick), new_X))
# new_X_padded  = list(map(lambda x: np.pad(x, (0, max_tick - x.size), 'constant'), new_X))
new_X_padded  = list(map(lambda x: np.resize(x, med_tick) if np.size(x) >= med_tick else np.pad(x, (0, med_tick - x.size), 'constant'), new_X))
X_train, X_test, y_train, y_test = train_test_split(new_X_padded, y, test_size=0.25, random_state=42)

""" snippet to plot sizes
sizer = []
for i in new_X:
    sizer.append(np.shape(i))
import matplotlib.pyplot as plt
plt.plot(sizer)
plt.show()
print(np.median(sizer))
print(np.mean(sizer))
"""
# In[]: Multiclassify labeler
X = np.concatenate((np.load('atomic_v3_1.npy', allow_pickle=True), 
                    np.load('atomic_v3_2.npy', allow_pickle=True), 
                    np.load('atomic_v3_3.npy', allow_pickle=True), 
                    np.load('atomic_v3_4.npy', allow_pickle=True), 
                    np.load('atomic_v3_5.npy', allow_pickle=True), 
                    np.load('atomic_v3_6.npy', allow_pickle=True), 
                    # np.load('atomic_v3_7.npy', allow_pickle=True), 
                    np.load('atomic_v3_8.npy', allow_pickle=True), 
                    np.load('atomic_v3_9.npy', allow_pickle=True),
                    np.load('atomic_v3_10.npy', allow_pickle=True)))

steamer = []
for i in X:
    steamer.append(str(i[0][0]) + i[0][1])
steamer = pd.DataFrame(steamer,columns=['steamid'])
steamer = steamer["steamid"].value_counts()

new_X = []
max_tick = 0
min_tick = np.inf
y = []


for inst in X:
    atomic_inst = inst[1]
    atomic_inst = np.delete(atomic_inst, np.arange(1, atomic_inst.size, 35))
    # low_tick = list(filter(lambda x: atomic_inst[x] < 30, np.arange(0, atomic_inst.size, 34)))
    # low_tick_index = []
    # for i in low_tick:
    #     low_tick_index += list(range(i, i + 34))
    # atomic_inst = np.delete(atomic_inst, low_tick_index)
    steam_id = inst[0][0]
    hero_name = inst[0][1]
    if atomic_inst.size < 1000 or atomic_inst.size > 80000:
        continue
    # max_tick = atomic_inst.size if atomic_inst.size > max_tick else max_tick
    # min_tick = atomic_inst.size if atomic_inst.size < min_tick else min_tick

    # if steam_id == 76561198134243802:
    #     if hero_name == "CDOTA_Unit_Hero_Puck":
    #         y.append(1)
    #     else: continue
    # # elif steam_id == 76561198078399948:
    # else:
    #     y.append(0)
    # # else: continue
    if steamer[str(steam_id) + hero_name] >= 40:
        # if (str(steam_id) + hero_name == "76561198173337033CDOTA_Unit_Hero_Chen"):
        y.append(steam_id)
            # y.append(1)
        # else:
            # y.append(0)
    else:
        # y.append(0)
        continue
    new_X.append(atomic_inst)

med_tick = 20000
# new_X_padded  = list(map(lambda x: np.resize(x, min_tick), new_X))
# new_X_padded  = list(map(lambda x: np.pad(x, (0, max_tick - x.size), 'constant'), new_X))
new_X_padded  = list(map(lambda x: np.resize(x, med_tick) if np.size(x) >= med_tick else np.pad(x, (0, med_tick - x.size), 'constant'), new_X))
X_train, X_test, y_train, y_test = train_test_split(new_X_padded, y, test_size=0.25, random_state=42)
"""
Snippet to see player id games
for i in steamer.index:
    if steamer[i] >= 40:
        print(i + str(steamer[i]))
"""

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
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=42, max_iter=200, class_weight='balanced').fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('The accuracy of the Logistic Regression is ', round(accuracy_score(prediction_rm, y_test)*100,2))
# print('The precision of the Logistic Regression is ', round(precision_score(prediction_rm, y_test, pos_label=1)*100,2))
# print('The recall of the Logistic Regression is ', round(recall_score(prediction_rm, y_test, pos_label=1)*100,2))
# print('The f1_score of the Logistic Regression is ', round(f1_score(prediction_rm, y_test, pos_label=1)*100,2))
print('The micro precision of the Logistic Regression is ', round(precision_score(prediction_rm, y_test, pos_label=1, average='micro')*100,2))
print('The micro recall of the Logistic Regression is ', round(recall_score(prediction_rm, y_test, pos_label=1, average='micro')*100,2))
print('The micro f1_score of the Logistic Regression is ', round(f1_score(prediction_rm, y_test, pos_label=1, average='micro')*100,2))
# print('The EER value of the Logistic Regression is ', round(calculate_eer(prediction_rm, y_test)*100,2))

result_rm=cross_val_score(clf, new_X_padded, y, cv=5,scoring='accuracy')
print('----------------------The cross validated accuracy score for Logistic Regression is:',round(result_rm.mean()*100,2))
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
