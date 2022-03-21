# In[1]: Headerimport

from matplotlib.pyplot import axis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data

from sklearn.model_selection import cross_validate #score evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
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
def train(model, model_name, steamer, X):
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

        # X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.25, random_state=42)
        new_X = preprocessing.StandardScaler().fit(new_X).transform(new_X)
        # new_X = preprocessing.RobustScaler().fit(new_X).transform(new_X)


        # clf = model.fit(X_train, y_train)

        result_rm=cross_validate(model, new_X, y, cv=5,scoring={'precision': 'precision', 'recall': 'recall', 'accuracy': 'accuracy', 'f1': 'f1', 'roc_auc': 'roc_auc', 'eer': make_scorer(calculate_eer)})
        result_dict["accuracy"].append(round(result_rm["test_accuracy"].mean()*100,2))
        result_dict["precision"].append(round(result_rm["test_precision"].mean()*100,2))
        result_dict["recall"].append(round(result_rm["test_recall"].mean()*100,2))
        result_dict["f1"].append(round(result_rm["test_f1"].mean()*100,2))
        result_dict["roc"].append(round(result_rm["test_roc_auc"].mean()*100,2))
        result_dict["eer"].append(result_rm["test_eer"].mean())
        #result_values[player, hero_name] = round(result_rm["test_f1"].mean()*100,2)
        result_values[player, hero_name] = [result_dict["accuracy"][-1], result_dict["precision"][-1], result_dict["recall"][-1] ,result_dict["f1"][-1] ,result_dict["roc"][-1] ,result_dict["eer"][-1]] 
        print("batch " + str(counter))
        counter += 1
    print(model_name + " results --------------------------------------------------------")
    print("accuracy: " + str(sum(result_dict["accuracy"]) / len(result_dict["accuracy"])))
    print("precision: " + str(sum(result_dict["precision"]) / len(result_dict["precision"])))
    print("recall: " + str(sum(result_dict["recall"]) / len(result_dict["recall"])))
    print("f1: " + str(sum(result_dict["f1"]) / len(result_dict["f1"])))
    print("roc: " + str(sum(result_dict["roc"]) / len(result_dict["roc"])))
    print("eer: " + str(sum(result_dict["eer"]) / len(result_dict["eer"])))
    print(result_values)
    return result_dict, result_values

# In[3]: Load complex atomic features

import glob, os

li = []

os.chdir(os.getcwd() + "/trainable_features_1/combo/")
counter = 1
for file in glob.glob("*"):
    df = pd.read_csv(file, index_col=None, header=0)
    li.append(df)
    if counter == 4000:
        break
    counter += 1


X = pd.concat(li, axis=0, ignore_index=True)
#X.drop(X[ X['Tick'] < 25 ].index, inplace=True)
#X = X[X.groupby("Steam_id")["Steam_id"].transform('count').lt(8990)]
#X = X[X.groupby("Steam_id")["Steam_id"].transform('count').ge(7921)]
steamer = X["Steam_id"].value_counts()
print(steamer[:75])

eval_df = pd.DataFrame(columns=["Hero", 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], index = steamer[:75].index)
# In[4]: Scale Data

#train(LogisticRegression(class_weight="balanced", max_iter = 1000), "Logistic Regression", steamer, X)
#train(RandomForestClassifier(class_weight="balanced"), "Random Forest", steamer, X)
#train(MLPClassifier(random_state=42, alpha=0.001, solver="lbfgs", max_iter=1000), "MLP Classifier lbfgs 1000", steamer, X)
#train(MLPClassifier(random_state=42, alpha=0.001, max_iter=1000), "MLP Classifier adam 1000", steamer, X)


for j in range(2,16):
    overall_result_dict = {}
    overall_result_dict["accuracy"] = []
    overall_result_dict["precision"] = []
    overall_result_dict["recall"] = []
    overall_result_dict["f1"] = []
    overall_result_dict["roc"] = []
    overall_result_dict["eer"] = []

    for i in range(int(50/j)):
        X_tmp = X[X.groupby("Steam_id")["Steam_id"].transform('count').le(steamer.iloc[i*j])]
        X_tmp = X_tmp[X_tmp.groupby("Steam_id")["Steam_id"].transform('count').ge(steamer.iloc[i*j + j-1])]
        steamer_tmp = X_tmp["Steam_id"].value_counts()
        result_ev_tmp, result_all_tmp = train(LogisticRegression(class_weight="balanced", max_iter = 1000), "Logistic Regression", steamer_tmp, X_tmp)
        for pl in result_all_tmp:
            eval_df.loc[pl[0]]["Hero"] = pl[1]
            eval_df.loc[pl[0]][j] = result_all_tmp[pl][3]
        overall_result_dict["accuracy"].append(result_all_tmp[pl][0])
        overall_result_dict["precision"].append(result_all_tmp[pl][1])
        overall_result_dict["recall"].append(result_all_tmp[pl][2])
        overall_result_dict["f1"].append(result_all_tmp[pl][3])
        overall_result_dict["roc"].append(result_all_tmp[pl][4])
        overall_result_dict["eer"].append(result_all_tmp[pl][5])

    print("Overall Logistic Regresion results for " + str(j) + " group  --------------------------------------------------------")
    print("accuracy: " + str(sum(overall_result_dict["accuracy"]) / len(overall_result_dict["accuracy"])))
    print("precision: " + str(sum(overall_result_dict["precision"]) / len(overall_result_dict["precision"])))
    print("recall: " + str(sum(overall_result_dict["recall"]) / len(overall_result_dict["recall"])))
    print("f1: " + str(sum(overall_result_dict["f1"]) / len(overall_result_dict["f1"])))
    print("roc: " + str(sum(overall_result_dict["roc"]) / len(overall_result_dict["roc"])))
    print("eer: " + str(sum(overall_result_dict["eer"]) / len(overall_result_dict["eer"])))

eval_df.to_csv("lr_overall.csv")

