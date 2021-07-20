# In[1]: Headerimport
from operator import mul
from numpy.lib.function_base import append, average
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data

from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_validate #score evaluation

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
                    np.load('atomic_v3_10.npy', allow_pickle=True)))

steamer = []
for i in X:
    steamer.append(str(i[0][0]) + i[0][1])
steamer = pd.DataFrame(steamer,columns=['steamid'])
steamer = steamer["steamid"].value_counts()

plausible_player = []
for i in steamer.index:
    if (steamer[i] >= 10):
        plausible_player.append(i)
import itertools
player_perm = list(itertools.permutations(plausible_player, 2))

counter = 1
result_dict = {}
result_dict["accuracy"] = []
result_dict["precision"] = []
result_dict["recall"] = []
result_dict["f1"] = []

for perm in player_perm:
    new_X = []
    max_tick = 0
    min_tick = np.inf
    y = []

    for inst in X:
        atomic_inst = inst[1]
        atomic_inst = np.delete(atomic_inst, np.arange(1, atomic_inst.size, 35))
        # low_tick = list(filter(lambda x: atomic_inst[x] < 30, np.arange(0, atomic_inst.size, 34)))
        # low_tick_index = []
        # for i in low_tick:
        #     low_tick_index += list(range(i, i + 34))
        # atomic_inst = np.delete(atomic_inst, low_tick_index)
        steam_id = inst[0][0]
        hero_name = inst[0][1]
        if atomic_inst.size < 1000 or atomic_inst.size > 80000:
            continue
        # max_tick = atomic_inst.size if atomic_inst.size > max_tick else max_tick
        # min_tick = atomic_inst.size if atomic_inst.size < min_tick else min_tick


        # if steamer[str(steam_id) + hero_name] >= 5:
        if (str(steam_id) + hero_name == perm[0]):
            y.append(1)
        elif (str(steam_id) + hero_name == perm[1]):
            y.append(0)
        else:
            continue
        new_X.append(atomic_inst)

    med_tick = 20000
    # new_X_padded  = list(map(lambda x: np.resize(x, min_tick), new_X))
    # new_X_padded  = list(map(lambda x: np.pad(x, (0, max_tick - x.size), 'constant'), new_X))
    new_X_padded  = list(map(lambda x: np.resize(x, med_tick) if np.size(x) >= med_tick else np.pad(x, (0, med_tick - x.size), 'constant'), new_X))
    X_train, X_test, y_train, y_test = train_test_split(new_X_padded, y, test_size=0.25, random_state=42)
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(random_state=42, max_iter=200).fit(X_train, y_train)
    prediction_rm=clf.predict(X_test)
    result_rm=cross_validate(clf, new_X_padded, y, cv=5,scoring=['precision', 'recall', 'accuracy', 'f1'])
    result_dict["accuracy"].append(round(result_rm["test_accuracy"].mean()*100,2))
    result_dict["precision"].append(round(result_rm["test_precision"].mean()*100,2))
    result_dict["recall"].append(round(result_rm["test_recall"].mean()*100,2))
    result_dict["f1"].append(round(result_rm["test_f1"].mean()*100,2))
    print("batch " + str(counter) + "/" + str(len(player_perm)))
    counter += 1

# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 4, sharey=True, tight_layout=True)

axs[0].hist(result_dict["accuracy"])
axs[0].set_title('Accuracy')
axs[0].axis(xmin=0,xmax=100)
axs[1].hist(result_dict["precision"])
axs[1].set_title('Precision')
axs[1].axis(xmin=0,xmax=100)
axs[2].hist(result_dict["recall"])
axs[2].set_title('Recall')
axs[2].axis(xmin=0,xmax=100)
axs[3].hist(result_dict["f1"])
axs[3].set_title('F1')
axs[3].axis(xmin=0,xmax=100)
plt.show()

# %%
