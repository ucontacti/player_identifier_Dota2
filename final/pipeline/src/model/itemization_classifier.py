from model.base_classifier import BaseClassifier, calculate_eer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split #for split the data
from sklearn.model_selection import cross_validate #score evaluation
from sklearn import preprocessing
from sklearn.metrics import make_scorer

import importlib
import json
spam_spec = importlib.util.find_spec("tensorflow")
found_Keras = spam_spec is not None
if found_Keras:
    from tensorflow.keras.layers import Dense, Sequential

spam_spec = importlib.util.find_spec("util.ESN")
found_ESN = spam_spec is not None
if found_ESN:
    from util.ESN import ESNTimeSeries, ESNVanilla


REPLAY_TRACKER_PATH = "../resources/replay_tracker.csv"


class ItemizationClassifier(BaseClassifier):
    """
    A class to train itemization features

    ...

    Attributes
    ----------
    num_of_players : int
        Number of target players to classify
    trainable_feature : Pandas Dataframe
        Movement features data
    clf : Scikit Model
        Chosen model for training
    model_name: str
        String value of the model name
    estimator: Scikit Estimator
        Trained estimator

    """

    def __init__(self, num_of_players) -> None:
        """ 
        Loads the data and number of target players. Also if ESN is available,
        tries to create representation feature

        Args:
            num_of_players (int): Number of top target players to classify,
            ordered by number of features
        """
        super().__init__(num_of_players)
        self.loaded_dict = json.load(open("../resources/item_converter.txt"))
        if found_ESN:
            self.__create_esn()

    def found_ESN(self) -> bool:
        return found_ESN
        
    def found_Keras(self) -> bool:
        return found_Keras

    def __create_esn(self) -> None:
        """ 
        Create fixed length ESN trainable fetures from raw itemization
        """
        replay_tracker = pd.read_csv(REPLAY_TRACKER_PATH)
        authentic_match_id = replay_tracker.loc[(replay_tracker['state'] == 7) & (replay_tracker['1_tick'] == True), 'replay_id'].tolist()
        
        input = []
        self.steam_id_list = []
        for match_id in authentic_match_id:
            match_id = str(match_id)
            path = "../features/" + match_id + "_item_change_tmp.csv"
            if not os.path.isfile(path):
                continue
            df_item = pd.read_csv(path)
            df_item.fillna(0, inplace=True)
            df_item.replace(self.loaded_dict, inplace=True)
            df_match_info = pd.read_csv("../features/"  + match_id + "_info.csv")
            dfs_item = [rows for _, rows in df_item.groupby('Hero')]
            for item in dfs_item:
                steam_id = df_match_info.loc[df_match_info["Hero"] == item["Hero"].iloc[0]].iloc[0]["SteamId"]
                hero_name = df_match_info.loc[df_match_info["Hero"] == item["Hero"].iloc[0]].iloc[0]["Hero"]
                temp_item = item[["Item1", "Item2", "Item3", "Item4", "Item5", "Item6", "Item7", "Item8", "Item9"]].values.astype(int)
                input.append(np.apply_along_axis(self.one_hot, 1, temp_item))
                # input.append(temp_item)
                self.steam_id_list.append(steam_id)
        self.steam_id_list = np.array(self.steam_id_list)
        X_TSObjs=[ESNTimeSeries(Xi) for Xi in input]

        numberOfReservoirNeurons=100
        esnObj=ESNVanilla(numberOfReservoirNeurons,n=X_TSObjs[0].inputDim,saveLastHiddenState=True)

        self.trainable_features=[esnObj.getLastInternalStateEmbedding(X_TSObjs_i) for X_TSObjs_i in X_TSObjs]
        
        self.trainable_features=np.array(self.trainable_features)

    
    def train_and_eval(self) -> None:
        """ 
        Trains loaded data based on the selected number of players
        """
        result_values = {}
        result_dict = {}
        result_dict["accuracy"] = []
        result_dict["precision"] = []
        result_dict["recall"] = []
        result_dict["f1"] = []
        result_dict["roc"] = []
        result_dict["eer"] = []
        
        steamer = pd.DataFrame(self.steam_id_list).value_counts()
        # print(steamer[:self.num_of_players])
        self.trainable_features = np.delete(self.trainable_features, np.where(~np.isin(self.steam_id_list, steamer[:self.num_of_players].index.to_list())), axis = 0)
        y = np.delete(self.steam_id_list, np.where(~np.isin(self.steam_id_list, steamer[:self.num_of_players].index.to_list())))
        self.trainable_features = preprocessing.StandardScaler().fit(self.trainable_features).transform(self.trainable_features)
        
        for counter, player in enumerate(steamer[:self.num_of_players].index):
            new_y = np.where(y == player, 1, 0)
            result_rm = cross_validate(self.clf, self.trainable_features, new_y, cv = 5,  \
                        scoring={'precision': 'precision', 'recall': 'recall', 'accuracy': 'accuracy', \
                        'f1': 'f1', 'roc_auc': 'roc_auc', 'eer': make_scorer(calculate_eer)}, \
                        return_estimator=True, return_train_score=True)
            result_dict["accuracy"].append(round(result_rm["test_accuracy"].mean()*100,2))
            result_dict["precision"].append(round(result_rm["test_precision"].mean()*100,2))
            result_dict["recall"].append(round(result_rm["test_recall"].mean()*100,2))
            result_dict["f1"].append(round(result_rm["test_f1"].mean()*100,2))
            result_dict["roc"].append(round(result_rm["test_roc_auc"].mean()*100,2))
            result_dict["eer"].append(result_rm["test_eer"].mean())
            #result_values[player, hero_name] = round(result_rm["test_f1"].mean()*100,2)
            print(f"batch {counter}")
            self.estimator = result_rm["estimator"]
        print(f"{self.model_name} results for --------------------------------------------------------")
        print(f"accuracy: {(sum(result_dict['accuracy']) / len(result_dict['accuracy']))}")
        print(f"precision: {(sum(result_dict['precision']) / len(result_dict['precision']))}")
        print(f"recall: {(sum(result_dict['recall']) / len(result_dict['recall']))}")
        print(f"f1: {(sum(result_dict['f1']) / len(result_dict['f1']))}")
        print(f"roc: {(sum(result_dict['roc']) / len(result_dict['roc']))}")
        print(f"eer: {(sum(result_dict['eer']) / len(result_dict['eer']))}")
        
        # return result_dict, result_values


    def one_hot(self, row):
        tmp_mask = np.zeros(len(self.loaded_dict) + 1)
        tmp_mask[row] = 1
        return tmp_mask

    
    def nn_train_and_eval(self) -> None:
        result_values = {}
        result_dict = {}
        result_dict["accuracy"] = []
        result_dict["precision"] = []
        result_dict["recall"] = []
        result_dict["f1"] = []
        result_dict["roc"] = []
        result_dict["eer"] = []
        
        steamer = pd.DataFrame(self.steam_id_list).value_counts()
        self.trainable_features = np.delete(self.trainable_features, np.where(~np.isin(self.steam_id_list, steamer[:self.num_of_players].index.to_list())), axis = 0)
        y = np.delete(self.steam_id_list, np.where(~np.isin(self.steam_id_list, steamer[:self.num_of_players].index.to_list())))
        self.trainable_features = preprocessing.StandardScaler().fit(self.trainable_features).transform(self.trainable_features)
        
        for counter, player in enumerate(steamer[:self.num_of_players].index):
            new_y = np.where(y == player, 1, 0)
            
            ev_accuracy = []
            ev_precision = []
            ev_recall = []
            ev_f1 = []

            skf = StratifiedKFold(n_splits=5, shuffle=False)
            for (train_index, test_index) in  skf.split(self.trainable_features, new_y):
                X_train, X_test = np.array(self.trainable_features)[train_index.astype(int)], np.array(self.trainable_features)[test_index.astype(int)]
                y_train, y_test = np.array(new_y)[train_index.astype(int)], np.array(new_y)[test_index.astype(int)]

                num_features = X_train.shape[1]
                
                super().define_nn_model(num_features)
                # compile the keras model
                self.clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
                self.clf.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=20)
                # evaluate the keras model
                # _, accuracy = model.evaluate(X_test, y_test)
                results = self.clf.evaluate(X_test, y_test, batch_size=100)
                ev_accuracy.append(results[1])
                ev_precision.append(results[2])
                ev_recall.append(results[3])
                if ((results[2] + results[3]) == 0):
                    ev_f1.append(0)
                else:
                    ev_f1.append(2 * results[2] * results[3] / (results[2] + results[3]))



            result_dict["accuracy"].append(round(sum(ev_accuracy) / len(ev_accuracy)*100,2))
            result_dict["precision"].append(round(sum(ev_precision) / len(ev_precision)*100,2))
            result_dict["recall"].append(round(sum(ev_recall) / len(ev_recall)*100,2))
            result_dict["f1"].append(round(sum(ev_f1) / len(ev_f1)*100,2))
            #result_dict["roc"].append(round(result_rm["test_roc_auc"].mean()*100,2))
            #result_dict["eer"].append(result_rm["test_eer"].mean())
            result_values[player, hero_name] = round(sum(ev_f1) / len(ev_f1)*100,2)
            print(f"batch {counter}")
            #self.estimator = result_rm["estimator"]
        print(f"{self.model_name} results for --------------------------------------------------------")
        print(f"accuracy: {(sum(result_dict['accuracy']) / len(result_dict['accuracy']))}")
        print(f"precision: {(sum(result_dict['precision']) / len(result_dict['precision']))}")
        print(f"recall: {(sum(result_dict['recall']) / len(result_dict['recall']))}")
        print(f"f1: {(sum(result_dict['f1']) / len(result_dict['f1']))}")
        print(f"roc: {(sum(result_dict['roc']) / len(result_dict['roc']))}")
        print(f"eer: {(sum(result_dict['eer']) / len(result_dict['eer']))}")
            
        # return result_dict, result_values
