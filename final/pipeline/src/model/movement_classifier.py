from model.base_classifier import BaseClassifier, calculate_eer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate #score evaluation
from sklearn import preprocessing
from glob import glob

from sklearn.metrics import make_scorer, roc_curve #for eer
from scipy.optimize import brentq #for eer
from scipy.interpolate import interp1d #for eer

import importlib
spam_spec = importlib.util.find_spec("tensorflow")
found_Keras = spam_spec is not None
if found_Keras:
    from tensorflow.keras.layers import Dense, Sequential


class MovementClassifier(BaseClassifier):
    """
    A class to train mouse movement features using

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
        Loads the data and number of target players

        Args:
            num_of_players (int): Number of top target players to classify,
            ordered by number of features
        """
        super().__init__(num_of_players)
        self.__load_movement_data()
    
    def found_Keras(self) -> bool:
        return found_Keras

    def __load_movement_data(self) -> None:
        """ 
        Internal data loaded method
        """
        tmp_df_holder = []
        for file in glob("../resources/trainable_features/combo/*.csv"):
            df = pd.read_csv(file, index_col=None, header=0)
            tmp_df_holder.append(df)
        self.trainable_feature = pd.concat(tmp_df_holder, axis=0, ignore_index=True)
        tmp_df_holder.clear()
        self.trainable_feature.fillna(0, inplace=True)
        
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
        
        drop_list = ["Label", "Hero", "Steam_id", "Cur_cr_min", "Cur_cr_max", "Cur_cr_mean", "Cur_cr_std"]
        steamer = self.trainable_feature["Steam_id"].value_counts()
        self.trainable_feature = self.trainable_feature[self.trainable_feature.groupby("Steam_id")["Steam_id"].transform('count').ge(steamer.iloc[self.num_of_players - 1])]
        target = self.trainable_feature["Steam_id"].value_counts()
        
        for counter, player in enumerate(target.index):
            self.trainable_feature.loc[self.trainable_feature["Steam_id"] == player, "Label"] = 1
            self.trainable_feature.loc[self.trainable_feature["Steam_id"] != player, "Label"] = 0
            y = self.trainable_feature["Label"]
            hero_name = self.trainable_feature.loc[self.trainable_feature["Steam_id"] == player, "Hero"].iloc[0]
            new_X = self.trainable_feature.drop(drop_list, axis = 1)

            new_X = preprocessing.StandardScaler().fit(new_X).transform(new_X)

            result_rm = cross_validate(self.clf, new_X, y, cv = 10,  \
                        scoring={'precision': 'precision', 'recall': 'recall', 'accuracy': 'accuracy', \
                        'f1': 'f1', 'roc_auc': 'roc_auc', 'eer': make_scorer(calculate_eer)}, \
                        return_estimator=True, return_train_score=True)
            result_dict["accuracy"].append(round(result_rm["test_accuracy"].mean()*100,2))
            result_dict["precision"].append(round(result_rm["test_precision"].mean()*100,2))
            result_dict["recall"].append(round(result_rm["test_recall"].mean()*100,2))
            result_dict["f1"].append(round(result_rm["test_f1"].mean()*100,2))
            result_dict["roc"].append(round(result_rm["test_roc_auc"].mean()*100,2))
            result_dict["eer"].append(result_rm["test_eer"].mean())
            result_values[player, hero_name] = round(result_rm["test_f1"].mean()*100,2)
            print(f"batch {counter}")
            self.estimator = result_rm["estimator"]
        print(f"{self.model_name} results for --------------------------------------------------------")
        print(f"accuracy: {(sum(result_dict['accuracy']) / len(result_dict['accuracy']))}")
        print(f"precision: {(sum(result_dict['precision']) / len(result_dict['precision']))}")
        print(f"recall: {(sum(result_dict['recall']) / len(result_dict['recall']))}")
        print(f"f1: {(sum(result_dict['f1']) / len(result_dict['f1']))}")
        print(f"roc: {(sum(result_dict['roc']) / len(result_dict['roc']))}")
        print(f"eer: {(sum(result_dict['eer']) / len(result_dict['eer']))}")
        # print(result_values)
        # return result_dict, result_values


    def nn_train_and_eval(self) -> None:
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
        
        drop_list = ["Label", "Hero", "Steam_id", "Cur_cr_min", "Cur_cr_max", "Cur_cr_mean", "Cur_cr_std"]
        steamer = self.trainable_feature["Steam_id"].value_counts()
        self.trainable_feature = self.trainable_feature[self.trainable_feature.groupby("Steam_id")["Steam_id"].transform('count').ge(steamer.iloc[self.num_of_players - 1])]
        target = self.trainable_feature["Steam_id"].value_counts()
        
        for counter, player in enumerate(target.index):
            self.trainable_feature.loc[self.trainable_feature["Steam_id"] == player, "Label"] = 1
            self.trainable_feature.loc[self.trainable_feature["Steam_id"] != player, "Label"] = 0
            y = self.trainable_feature["Label"]
            hero_name = self.trainable_feature.loc[self.trainable_feature["Steam_id"] == player, "Hero"].iloc[0]
            new_X = self.trainable_feature.drop(drop_list, axis = 1)

            new_X = preprocessing.StandardScaler().fit(new_X).transform(new_X)

            ev_accuracy = []
            ev_precision = []
            ev_recall = []
            ev_f1 = []
            skf = StratifiedKFold(n_splits=5, shuffle=False)
            for (train_index, test_index) in  skf.split(new_X, y):

                X_train, X_test = np.array(new_X)[train_index.astype(int)], np.array(new_X)[test_index.astype(int)]
                y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]

                num_features = X_train.shape[1]
                if self.model_numer == 5 or self.model_number == 6:
                    X_train = X_train.reshape(-1, 1, num_features)
                    X_test = X_test.reshape(-1, 1, num_features)


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
        # print(result_values)
        # return result_dict, result_values
