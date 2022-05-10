from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd
import os
from sklearn.model_selection import train_test_split #for split the data
from ESN import *
from sklearn.model_selection import cross_validate #score evaluation
from sklearn import preprocessing

# In[]: Calculate eer rate
from sklearn.metrics import make_scorer, roc_curve #for eer
from scipy.optimize import brentq #for eer
from scipy.interpolate import interp1d #for eer
def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


# In[]: Trainer
def train(model, model_name, X, labels):
    result_values = {}
    result_dict = {}
    result_dict["accuracy"] = []
    result_dict["precision"] = []
    result_dict["recall"] = []
    result_dict["f1"] = []
    result_dict["roc"] = []
    result_dict["eer"] = []
    steamer = pd.DataFrame(labels).value_counts()
    X = np.delete(X, np.where(~np.isin(labels, steamer[:2].index.to_list())), axis = 0)
    y = np.delete(labels, np.where(~np.isin(labels, steamer[:2].index.to_list())))
    X = preprocessing.StandardScaler().fit(X).transform(X)
    counter = 1
    for player in steamer[:2].index:
        new_y = np.where(y == player, 1, 0)

        # X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.25, random_state=42)
        # new_X = preprocessing.RobustScaler().fit(new_X).transform(new_X)

        # clf = model.fit(X_train, y_train)
        result_rm=cross_validate(model, X, new_y, cv=5,scoring={'precision': 'precision', 'recall': 'recall', 'accuracy': 'accuracy', 'f1': '>
        result_dict["accuracy"].append(round(result_rm["test_accuracy"].mean()*100,2))
        result_dict["precision"].append(round(result_rm["test_precision"].mean()*100,2))
        result_dict["recall"].append(round(result_rm["test_recall"].mean()*100,2))
        result_dict["f1"].append(round(result_rm["test_f1"].mean()*100,2))
        result_dict["roc"].append(round(result_rm["test_roc_auc"].mean()*100,2))
        result_dict["eer"].append(result_rm["test_eer"].mean())
        #result_values[player, hero_name] = round(result_rm["test_f1"].mean()*100,2)
        print("batch " + str(counter))
        counter += 1
    print(model_name + " results --------------------------------------------------------")
    print("accuracy: " + str(sum(result_dict["accuracy"]) / len(result_dict["accuracy"])))
    print("precision: " + str(sum(result_dict["precision"]) / len(result_dict["precision"])))
    print("recall: " + str(sum(result_dict["recall"]) / len(result_dict["recall"])))
    print("f1: " + str(sum(result_dict["f1"]) / len(result_dict["f1"])))
    print("roc: " + str(sum(result_dict["roc"]) / len(result_dict["roc"])))
    print("eer: " + str(sum(result_dict["eer"]) / len(result_dict["eer"])))
    #print(result_values)
    #return result_dict, result_values


replay_tracker = pd.read_csv("replay_tracker.csv")
authentic_match_id = replay_tracker.loc[(replay_tracker['state'] == 6) & (replay_tracker['1_tick'] == True), 'replay_id'].tolist()

input = []
steam_id_list = []
for match_id in authentic_match_id:
    #print(match_id)
    match_id = str(match_id)
    path = "../automation/features/" + match_id + "_item_change_tmp.csv"
    if not os.path.isfile(path):
        continue
    df_item = pd.read_csv(path)
    df_match_info = pd.read_csv("../automation/features/"  + match_id + "_info.csv")
    dfs_item = [rows for _, rows in df_item.groupby('Hero')]
    for item in dfs_item:
        steam_id = df_match_info.loc[df_match_info["Hero"] == item["Hero"].iloc[0]].iloc[0]["SteamId"]
        hero_name = df_match_info.loc[df_match_info["Hero"] == item["Hero"].iloc[0]].iloc[0]["Hero"]
        input.append(np.array(item[["Item1", "Item2", "Item3", "Item4", "Item5", "Item6", "Item7", "Item8", "Item9"]].values.tolist()))
        steam_id_list.append(steam_id)
steam_id_list = np.array(steam_id_list)
X_TSObjs=[ESNTimeSeries(Xi) for Xi in input]

numberOfReservoirNeurons=100
esnObj=ESNVanilla(numberOfReservoirNeurons,n=X_TSObjs[0].inputDim,saveLastHiddenState=False)

X_H=[esnObj.getLastInternalStateEmbedding(X_TSObjs_i) for X_TSObjs_i in X_TSObjs]
X_H=np.array(X_H)
#print(X_H.shape)

train(LogisticRegression(class_weight="balanced", max_iter = 1000), "Logistic Regression", X_H, steam_id_list)




