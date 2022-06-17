# In[1]: Headerimport

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data

from sklearn.model_selection import cross_validate #score evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate #score evaluation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.constraints import max_norm
import tensorflow as tf
from sklearn import preprocessing
#from tensorflow.keras.layers.normalization import BatchNormalization

# In[]: Calculate eer rate
from sklearn.metrics import make_scorer, roc_curve #for eer
from scipy.optimize import brentq #for eer
from scipy.interpolate import interp1d #for eer
def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


# In[]: Trainer
def train(model_name, steamer, X):
    result_values = {}
    result_dict = {}
    result_dict["accuracy"] = []
    result_dict["precision"] = []
    result_dict["recall"] = []
    result_dict["f1"] = []
    result_dict["roc"] = []
    result_dict["eer"] = []
    counter = 1
    for player in steamer.index:
        X.loc[X["Steam_id"] == player, "Label"] = 1
        X.loc[X["Steam_id"] != player, "Label"] = 0
        y = X["Label"]
        hero_name = X.loc[X["Steam_id"] == player, "Hero"].iloc[0]
        new_X = X.drop(["Label", "Hero", "Steam_id"], axis = 1)

        #X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.25, random_state=42)
        new_X = preprocessing.StandardScaler().fit(new_X).transform(new_X)
        #clf = model.fit(X_train, y_train)
        #result_rm=cross_validate(clf, new_X, y, cv=5,scoring={'precision': 'precision', 'recall': 'recall', 'accuracy': 'accuracy', 'f1': 'f1', 'roc_auc': 'roc_auc', 'eer': make_scorer(calculate_eer)})
        
        ev_accuracy = []
        ev_precision = []
        ev_recall = []
        ev_f1 = []
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        for (train_index, test_index) in  skf.split(new_X, y):

            X_train, X_test = np.array(new_X)[train_index.astype(int)], np.array(new_X)[test_index.astype(int)]
            y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]

            model = Sequential()
            # model.add(Dense(50, input_dim=34, activation='relu', kernel_constraint=maxnorm(3)))
            print(X_train.shape)
            model.add(Dense(100, input_dim=35, activation='relu'))
            #model.add(Dense(100, activation='relu', kernel_constraint=max_norm(3)))
            model.add(Dense(100, activation='relu'))
            #model.add(Dropout(0.2))
            model.add(Dense(50, activation='relu'))
            model.add(Dense(20, activation='relu'))
            model.add(Dense(100, activation='relu'))
            #model.add(Dropout(0.2))
            model.add(Dense(50, activation='relu'))
            # model.add(Dense(100, activation='relu', kernel_constraint=maxnorm(3)))
            model.add(Dense(1, activation='sigmoid'))
            # compile the keras model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
            # fit the keras model on the dataset
            model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=20)
            # evaluate the keras model
            # _, accuracy = model.evaluate(X_test, y_test)
            results = model.evaluate(X_test, y_test, batch_size=100)
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

        print("batch " + str(counter))
        counter += 1
        if counter == 6:
             break
    print(model_name + " results --------------------------------------------------------")
    print("accuracy: " + str(sum(result_dict["accuracy"]) / len(result_dict["accuracy"])))
    print("precision: " + str(sum(result_dict["precision"]) / len(result_dict["precision"])))
    print("recall: " + str(sum(result_dict["recall"]) / len(result_dict["recall"])))
    print("f1: " + str(sum(result_dict["f1"]) / len(result_dict["f1"])))
    #print("roc: " + str(sum(result_dict["roc"]) / len(result_dict["roc"])))
    #print("eer: " + str(sum(result_dict["eer"]) / len(result_dict["eer"])))
    print(result_values)
    return result_dict, result_values

# In[3]: Load complex atomic features

import glob, os

li = []

files = glob.glob("trainable_features_1/combo/pickle*")
for f in files:
    df = pd.read_pickle(f)
    df.drop(df[ df['Tick'] < 200 ].index, inplace=True)
    df.drop(df[ df['Tick'] > 2000 ].index, inplace=True)
    #df = pd.get_dummies(df,prefix=['Action'], columns = ['Action'])
    li.append(df)
X = pd.concat(li, axis=0, ignore_index=True)
li.clear()

#X.drop(X[ X['Tick'] < 8 ].index, inplace=True)
X.fillna(0, inplace=True)
steamer = X["Steam_id"].value_counts()
X = X[X.groupby("Steam_id")["Steam_id"].transform('count').ge(steamer.iloc[5])]
steamer = X["Steam_id"].value_counts()

train("Neural Network", steamer, X)
