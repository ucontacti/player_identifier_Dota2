# In[1]: Headerimport
from operator import mul
from numpy.core.shape_base import block
from numpy.lib.function_base import append, average
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate #score evaluation
from keras.models import Sequential
from keras.layers import Dense
import keras

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
                    np.load('atomic_v3_10.npy', allow_pickle=True),
                    np.load('atomic_v3_11.npy', allow_pickle=True),
                    np.load('atomic_v3_12.npy', allow_pickle=True),
                    np.load('atomic_v3_13.npy', allow_pickle=True),
                    np.load('atomic_v3_14.npy', allow_pickle=True)))


steamer = []
for i in X:
    steamer.append(str(i[0][0]) + i[0][1])
steamer = pd.DataFrame(steamer,columns=['steamid'])
steamer = steamer["steamid"].value_counts()
result_values = {}
result_dict = {}
result_dict["accuracy"] = []
result_dict["precision"] = []
result_dict["recall"] = []
result_dict["f1"] = []
counter = 1

for player in steamer.index:
    if steamer[player] >= 40:
        new_X = []
        max_tick = 0
        min_tick = np.inf
        y = []

        for inst in X:
            atomic_inst = inst[1]
            if (atomic_inst.size == 0):
                continue
            atomic_inst = np.delete(atomic_inst, np.arange(1, atomic_inst.size, 35))
            atomic_arr = []
            atomic_arr.append(np.mean(atomic_inst[np.arange(0, atomic_inst.size, 34)]))
            for i in range(1,33):
                if (i%4 == 1):
                    atomic_arr.append(np.min(atomic_inst[np.arange(i, atomic_inst.size, 34)]))
                elif (i%4 == 2):
                    atomic_arr.append(np.max(atomic_inst[np.arange(i, atomic_inst.size, 34)]))
                elif (i%4 == 3):
                    atomic_arr.append(np.mean(atomic_inst[np.arange(i, atomic_inst.size, 34)]))
                elif (i%4 == 0):
                    atomic_arr.append(np.std(atomic_inst[np.arange(i, atomic_inst.size, 34)]))
            atomic_arr.append(np.mean(atomic_inst[np.arange(33, atomic_inst.size, 34)]))
            atomic_arr = np.array(atomic_arr)
            atomic_arr = np.nan_to_num(atomic_arr, nan=0, posinf=0, neginf=0)
            
            steam_id = inst[0][0]
            hero_name = inst[0][1]
            if steamer[str(steam_id) + hero_name] >= 15:
                if (str(steam_id) + hero_name == player):
                    y.append(1)
                else:
                    y.append(0)
            else:
                continue
            new_X.append(atomic_arr)

        X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.25, random_state=42)
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

        result_rm=cross_validate(clf, new_X, y, cv=5,scoring=['precision', 'recall', 'accuracy', 'f1'])
        result_dict["accuracy"].append(round(result_rm["test_accuracy"].mean()*100,2))
        result_dict["precision"].append(round(result_rm["test_precision"].mean()*100,2))
        result_dict["recall"].append(round(result_rm["test_recall"].mean()*100,2))
        result_dict["f1"].append(round(result_rm["test_f1"].mean()*100,2))
        result_values[player] = round(result_rm["test_f1"].mean()*100,2)
        print("batch " + str(counter))
        counter += 1

# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 4, sharey=True, tight_layout=True)

axs[0].hist(result_dict["accuracy"], density=True)
axs[0].set_title('Accuracy')
axs[0].axis(xmin=0,xmax=100)
axs[1].hist(result_dict["precision"], density=True)
axs[1].set_title('Precision')
axs[1].axis(xmin=0,xmax=100)
axs[2].hist(result_dict["recall"], density=True)
axs[2].set_title('Recall')
axs[2].axis(xmin=0,xmax=100)
axs[3].hist(result_dict["f1"], density=True)
axs[3].set_title('F1')
axs[3].axis(xmin=0,xmax=100)
plt.savefig('dt_histo_6.png')
# %%
