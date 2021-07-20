# In[1]: Headerimport
from operator import mul
from numpy.lib.function_base import append, average
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data

from sklearn.model_selection import KFold #for K-fold cross validation
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
                    np.load('atomic_v3_10.npy', allow_pickle=True)))

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

    if steamer[str(steam_id) + hero_name] >= 10:
        if (str(steam_id) + hero_name == "76561198135593836CDOTA_Unit_Hero_Meepo"):
            # y.append(steam_id)
            y.append(1)
        # elif (str(steam_id) + hero_name == "76561198135593836CDOTA_Unit_Hero_Meepo"):
        else:
            y.append(0)
    else:
            # y.append(0)
        continue
    new_X.append(atomic_inst)

med_tick = 20000
# new_X_padded  = list(map(lambda x: np.resize(x, min_tick), new_X))
# new_X_padded  = list(map(lambda x: np.pad(x, (0, max_tick - x.size), 'constant'), new_X))
new_X_padded  = list(map(lambda x: list(np.resize(x, med_tick)) if np.size(x) >= med_tick else list(np.pad(x, (0, med_tick - x.size), 'constant')), new_X))
X_train, X_test, y_train, y_test = train_test_split(new_X_padded, y, test_size=0.25, random_state=42)

# In[]: Keras part 


from keras.models import Sequential
from keras.layers import Dense
import keras
model = Sequential()
model.add(Dense(50, input_dim=20000, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=150, validation_data=(X_test, y_test), batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X_test, y_test)
print('precision: %.2f' % (accuracy*100))
# %%