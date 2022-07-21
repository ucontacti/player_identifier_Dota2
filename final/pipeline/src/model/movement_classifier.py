from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data

from sklearn.model_selection import cross_validate #score evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from glob import glob

from sklearn.metrics import make_scorer, roc_curve #for eer
from scipy.optimize import brentq #for eer
from scipy.interpolate import interp1d #for eer

class MovementClassifier:
    """
    A class to train mouse movement features using simple Scikit linear models

    ...

    Attributes
    ----------
    num_of_players : int
        Number of target players to classify
    data : Pandas Dataframe
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
        self.num_of_players = num_of_players
        self.__load_movement_data()

    def select_model(self, model_number: int) -> None:
        """ 
        Args:
            model_number (int): Given model number in commandline
        """
        if model_number == 1:
            self.clf = LogisticRegression()
            self.model_name = "Logistic Regression"
        elif model_number == 2:
            self.clf = RandomForestClassifier()
            self.model_name = "Random Forest"
        elif model_number == 3:
            self.clf = DecisionTreeClassifier()
            self.model_name = "Decision Tree"
        else:
            pass
        
    def train_and_eval(self) -> None:
        """ 
        Trains loaded data based on the selected number of players
        """
        drop_list = ["Label", "Hero", "Steam_id", "Cur_cr_min", "Cur_cr_max", "Cur_cr_mean", "Cur_cr_std"]
        target = self.data["Steam_id"].value_counts()
        result_values = {}
        result_dict = {}
        result_dict["accuracy"] = []
        result_dict["precision"] = []
        result_dict["recall"] = []
        result_dict["f1"] = []
        result_dict["roc"] = []
        result_dict["eer"] = []
        for counter, player in enumerate(target.index):
            self.data.loc[self.data["Steam_id"] == player, "Label"] = 1
            self.data.loc[self.data["Steam_id"] != player, "Label"] = 0
            y = self.data["Label"]
            hero_name = self.data.loc[self.data["Steam_id"] == player, "Hero"].iloc[0]
            new_X = self.data.drop(drop_list, axis = 1)

            new_X = preprocessing.StandardScaler().fit(new_X).transform(new_X)

            result_rm = cross_validate(self.clf, new_X, y, cv = 10,  \
                        scoring={'precision': 'precision', 'recall': 'recall', 'accuracy': 'accuracy', \
                        'f1': 'f1', 'roc_auc': 'roc_auc', 'eer': make_scorer(self.__calculate_eer)}, \
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


    def __load_movement_data(self) -> None:
        """ 
        Internal data loaded method
        """
        tmp_df_holder = []
        for file in glob("../resources/trainable_features/combo/*.csv"):
            df = pd.read_csv(file, index_col=None, header=0)
            tmp_df_holder.append(df)
        self.data = pd.concat(tmp_df_holder, axis=0, ignore_index=True)
        tmp_df_holder.clear()
        self.data.fillna(0, inplace=True)
        steamer = self.data["Steam_id"].value_counts()
        self.data = self.data[self.data.groupby("Steam_id")["Steam_id"].transform('count').ge(steamer.iloc[self.num_of_players - 1])]

    def __calculate_eer(self, y_true, y_score):
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer

    def calculate_coefficiency():
        pass

    def plot_data():
        pass

    def save_model():
        pass