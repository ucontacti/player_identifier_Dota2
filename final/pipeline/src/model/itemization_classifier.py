from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd
import os
from sklearn.model_selection import train_test_split #for split the data
from sklearn.model_selection import cross_validate #score evaluation
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import make_scorer, roc_curve #for eer
from scipy.optimize import brentq #for eer
from scipy.interpolate import interp1d #for eer

import importlib
spam_spec = importlib.util.find_spec("util.ESN")
found_ESN = spam_spec is not None
if found_ESN:
    from util.ESN import ESNTimeSeries, ESNVanilla


REPLAY_TRACKER_PATH = "../resources/replay_tracker.csv"


class ItemizationClassifier:
    def __init__(self, num_of_players) -> None:
        self.num_of_players = num_of_players
        if found_ESN:
            self.__create_esn()

    def found_ESN(self) -> bool:
        return found_ESN

    def select_model(self, model_name: int) -> None:
        if model_name == 1:
            self.clf = LogisticRegression()
            self.model_name = "Logistic Regression"
        elif model_name == 2:
            self.clf = RandomForestClassifier()
            self.model_name = "Random Forest"
        elif model_name == 3:
            self.clf = DecisionTreeClassifier()
            self.model_name = "Decision Tree"
        else:
            pass
        
    def __create_esn(self) -> None:
        replay_tracker = pd.read_csv(REPLAY_TRACKER_PATH)
        authentic_match_id = replay_tracker.loc[(replay_tracker['state'] == 6) & (replay_tracker['1_tick'] == True), 'replay_id'].tolist()

        input = []
        self.steam_id_list = []
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
                self.steam_id_list.append(steam_id)
        steam_id_list = np.array(self.steam_id_list)
        X_TSObjs=[ESNTimeSeries(Xi) for Xi in input]

        numberOfReservoirNeurons=100
        esnObj=ESNVanilla(numberOfReservoirNeurons,n=X_TSObjs[0].inputDim,saveLastHiddenState=False)

        self.X_H=[esnObj.getLastInternalStateEmbedding(X_TSObjs_i) for X_TSObjs_i in X_TSObjs]
        self.X_H=np.array(self.X_H)


    def __calculate_eer(y_true, y_score):
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer
    
    def train_and_eval(self, target) -> None:
        result_values = {}
        result_dict = {}
        result_dict["accuracy"] = []
        result_dict["precision"] = []
        result_dict["recall"] = []
        result_dict["f1"] = []
        result_dict["roc"] = []
        result_dict["eer"] = []
        steamer = pd.DataFrame(self.steam_id_list).value_counts()
        self.X_H = np.delete(self.X_H, np.where(~np.isin(self.steam_id_list, target)), axis = 0)
        y = np.delete(self.steam_id_list, np.where(~np.isin(self.steam_id_list, target)))
        self.X_H = preprocessing.StandardScaler().fit(self.X_H).transform(self.X_H)
        for counter, player in enumerate(target):
            new_y = np.where(y == player, 1, 0)
            result_rm = cross_validate(self.clf, self.X_H, new_y, cv = 5,  \
                        scoring={'precision': 'precision', 'recall': 'recall', 'accuracy': 'accuracy', \
                        'f1': 'f1', 'roc_auc': 'roc_auc', 'eer': make_scorer(self.__calculate_eer)}, \
                        return_estimator=True, return_train_score=True)
            result_dict["accuracy"].append(round(result_rm["test_accuracy"].mean()*100,2))
            result_dict["precision"].append(round(result_rm["test_precision"].mean()*100,2))
            result_dict["recall"].append(round(result_rm["test_recall"].mean()*100,2))
            result_dict["f1"].append(round(result_rm["test_f1"].mean()*100,2))
            result_dict["roc"].append(round(result_rm["test_roc_auc"].mean()*100,2))
            result_dict["eer"].append(result_rm["test_eer"].mean())
            #result_values[player, hero_name] = round(result_rm["test_f1"].mean()*100,2)
            print("batch " + str(counter))
        # print(self.model_name + " results --------------------------------------------------------")
        print("accuracy: " + str(sum(result_dict["accuracy"]) / len(result_dict["accuracy"])))
        print("precision: " + str(sum(result_dict["precision"]) / len(result_dict["precision"])))
        print("recall: " + str(sum(result_dict["recall"]) / len(result_dict["recall"])))
        print("f1: " + str(sum(result_dict["f1"]) / len(result_dict["f1"])))
        print("roc: " + str(sum(result_dict["roc"]) / len(result_dict["roc"])))
        print("eer: " + str(sum(result_dict["eer"]) / len(result_dict["eer"])))
            
        # return result_dict, result_values

    def __preprocess(self) -> None:
        self.data = preprocessing.StandardScaler().fit(self.data).transform(self.data)

    def calculate_coefficiency():
        pass

    def plot_data():
        pass

    def save_model():
        pass