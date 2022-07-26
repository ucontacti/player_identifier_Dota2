from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import make_scorer, roc_curve #for eer
from scipy.optimize import brentq #for eer
from scipy.interpolate import interp1d #for eer

import importlib
spam_spec = importlib.util.find_spec("tensorflow")
found_Keras = spam_spec is not None
if found_Keras:
    from tensorflow.keras.layers import Dense, Sequential, LSTM, GRU

def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


class BaseClassifier:
    def __init__(self, num_of_players) -> None:
        self.num_of_players = num_of_players    

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
        elif model_number == 4:    
            self.model_name = "Shallow Neural Network"
        elif model_number == 5:    
            self.model_name = "LSTM Network"
        elif model_number == 6:    
            self.model_name = "GRU Neural Network"
        else:
            pass
        self.model_number = model_number
            
    def define_nn_model(self, num_features):
        self.clf = Sequential()
        
        if self.model_number == 4:
            self.clf.add(Dense(100, input_dim=num_features, activation='relu'))
        elif self.model_number == 5:
            self.clf.add(LSTM(100, input_shape=(1, num_features)))
        elif self.model_number == 6:
            self.clf.add(GRU(100, input_shape=(1, num_features)))
        
        self.clf.add(Dense(1, activation='sigmoid'))        

    def calculate_coefficiency():
        pass

    def plot_data():
        pass

    def save_model():
        pass