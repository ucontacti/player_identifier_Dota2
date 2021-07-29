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
                    np.load('atomic_v3_10.npy', allow_pickle=True)))

steamer = []
for i in X:
    steamer.append(str(i[0][0]) + i[0][1])
steamer = pd.DataFrame(steamer,columns=['steamid'])
steamer = steamer["steamid"].value_counts()

result_dict = {}
result_dict["accuracy"] = []
result_dict["precision"] = []
result_dict["recall"] = []
result_dict["f1"] = []
counter = 1

for player in steamer.index:
    if steamer[player] >= 25:
        new_X = []
        max_tick = 0
        min_tick = np.inf
        y = []

        for inst in X:
            atomic_inst = inst[1]
            atomic_inst = np.delete(atomic_inst, np.arange(1, atomic_inst.size, 35))
            steam_id = inst[0][0]
            hero_name = inst[0][1]
            if atomic_inst.size < 1000 or atomic_inst.size > 80000:
                continue
            if steamer[str(steam_id) + hero_name] >= 10:
                if (str(steam_id) + hero_name == player):
                    y.append(1)
                else:
                    y.append(0)
            else:
                continue
            new_X.append(atomic_inst)

        med_tick = 20000
        new_X_padded  = list(map(lambda x: np.resize(x, med_tick) if np.size(x) >= med_tick else np.pad(x, (0, med_tick - x.size), 'constant'), new_X))

        y = np.array(y)

        ev_accuracy = []
        ev_precision = []
        ev_recall = []
        ev_f1 = []
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        for (train_index, test_index) in  skf.split(new_X_padded, y):

            X_train, X_test = np.array(new_X_padded)[train_index.astype(int)], np.array(new_X_padded)[test_index.astype(int)]
            y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]

            model = Sequential()
            model.add(Dense(50, input_dim=20000, activation='relu'))
            model.add(Dense(50, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            # compile the keras model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
            # fit the keras model on the dataset
            model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=20)
            # evaluate the keras model
            # _, accuracy = model.evaluate(X_test, y_test)
            results = model.evaluate(X_test, y_test, batch_size=40)
            ev_accuracy.append(results[1])
            ev_precision.append(results[2])
            ev_recall.append(results[3])
            if ((results[2] + results[3]) == 0):
                ev_f1.append(0)
            else:
                ev_f1.append(2 * results[2] * results[3] / (results[2] + results[3]))

        # X_train, X_test, y_train, y_test = train_test_split(new_X_padded, y, test_size=0.25, random_state=42)
        # from sklearn.linear_model import LogisticRegression

        # clf = LogisticRegression(random_state=42, max_iter=200).fit(X_train, y_train)
        # prediction_rm=clf.predict(X_test)
        # result_rm=cross_validate(clf, new_X_padded, y, cv=5,scoring=['precision', 'recall', 'accuracy', 'f1'])
        result_dict["accuracy"].append(round(sum(ev_accuracy) / len(ev_accuracy)*100,2))
        result_dict["precision"].append(round(sum(ev_precision) / len(ev_precision)*100,2))
        result_dict["recall"].append(round(sum(ev_recall) / len(ev_recall)*100,2))
        result_dict["f1"].append(round(sum(ev_f1) / len(ev_f1)*100,2))
        print("batch " + str(counter))
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
plt.savefig('nn_histo_2.png')
