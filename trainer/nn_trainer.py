# In[1]: Headerimport
from operator import mul
from numpy.lib.function_base import append, average
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate #score evaluation
# In[3]: Trim data to our need

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

new_X = []
max_tick = 0
min_tick = np.inf
y = []


for inst in X:
    atomic_inst = inst[1]
    atomic_inst = np.delete(atomic_inst, np.arange(1, atomic_inst.size, 35))
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
    # max_tick = atomic_inst.size if atomic_inst.size > max_tick else max_tick
    # min_tick = atomic_inst.size if atomic_inst.size < min_tick else min_tick

    if steamer[str(steam_id) + hero_name] >= 15:
        if (str(steam_id) + hero_name == "76561198162034645CDOTA_Unit_Hero_Dawnbreaker"):
        # y.append(steam_id)
            y.append(1)
        # elif (str(steam_id) + hero_name == "76561198135593836CDOTA_Unit_Hero_Meepo"):
        else:
            y.append(0)
    else:
            # y.append(0)
        continue
    new_X.append(atomic_inst)

# X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.25, random_state=42)

# In[]: Keras part 
from keras.models import Sequential
from keras.layers import Dense
import keras
y = np.array(y)

ev_accuracy = []
ev_precision = []
ev_recall = []
ev_f1 = []
skf = StratifiedKFold(n_splits=5, shuffle=False)
for (train_index, test_index) in  skf.split(new_X, y):

    X_train, X_test = np.array(new_X)[train_index.astype(int)], np.array(new_X)[test_index.astype(int)]
    y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]

    model = Sequential()
    model.add(Dense(50, input_dim=34, activation='relu'))
    model.add(Dense(50, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
    # fit the keras model on the dataset
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=20)
    # evaluate the keras model
    # _, accuracy = model.evaluate(X_test, y_test)
    results = model.evaluate(X_test, y_test, batch_size=40)
    ev_accuracy.append(results[1])
    # ev_precision.append(results[2])
    # ev_recall.append(results[3])
    # ev_f1.append(2 * results[2] * results[3] / (results[2] + results[3]))

    print('----------------------The cross validated accuracy score for Logistic Regression is:',round(results[1]*100,2))
    print('----------------------The cross validated precision score for Logistic Regression is:',round(results[2]*100,2))
    print('----------------------The cross validated recall score for Logistic Regression is:',round(results[3]*100,2))
    print('----------------------The cross validated f1 score for Logistic Regression is:',round(2 * results[2] * results[3] / (results[2] + results[3])*100,2))

# %%
