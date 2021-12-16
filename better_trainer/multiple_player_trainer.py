# In[1]: Headerimport

from matplotlib.pyplot import axis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data

from sklearn.model_selection import cross_validate #score evaluation

# In[3]: Load complex atomic features

import glob, os

li = []

os.chdir(os.getcwd() + "/trainable_features/combo/")
for file in glob.glob("*.csv"):
    df = pd.read_csv(file, index_col=None, header=0)
    li.append(df)


X = pd.concat(li, axis=0, ignore_index=True)
X = X[X.groupby("Steam_id")["Steam_id"].transform('count').ge(9000)]

steamer = X["Steam_id"].value_counts()

result_values = {}
result_dict = {}
result_dict["accuracy"] = []
result_dict["precision"] = []
result_dict["recall"] = []
result_dict["f1"] = []
counter = 1

for player in steamer.index:
    X.loc[X["Steam_id"] == player, "Label"] = 1
    X.loc[X["Steam_id"] != player, "Label"] = 0
    y = X["Label"]
    new_X = X.drop(["Label", "Hero", "Steam_id"], axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.25, random_state=42)

    # from sklearn.linear_model import LogisticRegression
    # clf = LogisticRegression(class_weight="balanced", max_iter = 1000).fit(X_train, y_train)

    # from sklearn.neural_network import MLPClassifier
    # clf = MLPClassifier(random_state=42, alpha=0.001).fit(X_train, y_train)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(class_weight="balanced").fit(X_train, y_train)
        
    result_rm=cross_validate(clf, new_X, y, cv=5,scoring=['precision', 'recall', 'accuracy', 'f1'])
    result_dict["accuracy"].append(round(result_rm["test_accuracy"].mean()*100,2))
    result_dict["precision"].append(round(result_rm["test_precision"].mean()*100,2))
    result_dict["recall"].append(round(result_rm["test_recall"].mean()*100,2))
    result_dict["f1"].append(round(result_rm["test_f1"].mean()*100,2))
    result_values[player] = round(result_rm["test_f1"].mean()*100,2)
    print("batch " + str(counter))
    counter += 1

print(sum(result_dict["accuracy"]) / len(result_dict["accuracy"]))
print(sum(result_dict["precision"]) / len(result_dict["precision"]))
print(sum(result_dict["recall"]) / len(result_dict["recall"]))
print(sum(result_dict["f1"]) / len(result_dict["f1"]))
print(result_values)