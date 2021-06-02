# In[1]: Headerimport
from numpy.lib.function_base import average
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data


# In[1]: test
df = pd.DataFrame({'time':[0,1,2], 'x':[1,2,4], 'y':[1,2,4]})
df["v_x"] = df["x"].diff() / df["time"].diff()
df["v_y"] = df["y"].diff() / df["time"].diff()
df["v"] = (df["v_x"]**2 + df["v_y"]**2)**(1/2)
df["a"] = df["v"].diff() / df["time"].diff()
df["jer"] = df["a"].diff() / df["time"].diff()
df["aom"] = np.arctan(df["x"].diff() / df["y"].diff())
df["aom"] = df["aom"].cumsum()
df["ang_v"] = df["aom"].diff() / df["time"].diff()
df = df.fillna(0)
print(df)



# In[2]: Read data and split train and test data

# authentic_match_id = [
#         "5910973712", 
#         "5911045449", 
#         "5912945803", 
#         "5913000523", 
#         "5915392078", 
#         "5926851979",
#         "5910938862",
#         "5911183098",
#         "5913013389",
#         "5913161793",
#         "5913253454",
#         "5915090628",
#         "5915090081",
#         "5915136220",
#         "5915141664",
#         "5897360586",
#         "5895383034",
#         "5891679036",
#         "5890067094",
#         "5886259086"
#     ]

authentic_match_id = [
        "5886259086",
        "5911279219",
        "5913341774",
        "5915287855",
        "5915824015",
        "5917145096",
        "5918982217",
        "5922226336",
        "5924712320",
        "5890067094",
        "5911353934",
        "5913415296",
        "5915332620",
        "5915833606",
        "5917193218",
        "5919074158",
        "5922293044",
        "5924812056",
        "5891679036",
        "5911377278",
        "5913525580",
        "5915342620",
        "5915868132",
        "5917211403",
        "5919165162",
        "5922379061",
        "5924956181",
        "5895383034",
        "5911457723",
        "5913646124",
        "5915345931",
        "5916888278",
        "5917225412",
        "5919286556",
        "5922438740",
        "5925119996",
        "5897360586",
        "5911478535",
        "5913776533",
        "5915392078",
        "5916889731",
        "5917274351",
        "5919422016",
        "5922512609",
        "5926065815",
        "5910801480",
        "5912941693",
        "5915088482",
        "5915422921",
        "5916890798",
        "5917291859",
        "5920377449",
        "5922602670",
        "5926148072",
        "5910802032",
        "5912945803",
        "5915090081",
        "5915448730",
        "5916941505",
        "5917291951",
        "5920449253",
        "5922679457",
        "5926212113",
        "5910870410",
        "5913000523",
        "5915090628",
        "5915536193",
        "5916945544",
        "5917346943",
        "5920494710",
        "5922768273",
        "5910896035",
        "5913013389",
        "5915136220",
        "5915536982",
        "5916953224",
        "5917363866",
        "5920564188",
        "5922871615",
        "5926402991",
        "5910938862",
        "5913070678",
        "5915141664",
        "5915537315",
        "5916999615",
        "5917380765",
        "5920638214",
        "5924068428",
        "5926498037",
        "5910973712",
        "5913118125",
        "5915155650",
        "5915595819",
        "5916999738",
        "5917443590",
        "5920714068",
        "5924160626",
        "5926596132",
        "5911032765",
        "5913152001",
        "5915204336",
        "5915613886",
        "5917003906",
        "5917514439",
        "5920806694",
        "5924231037",
        "5926732118",
        "5911045449",
        "5913161793",
        "5915209811",
        "5915624630",
        "5917058856",
        "5918691827",
        "5920889352",
        "5924343990",
        "5926851979",
        "5911144067",
        "5913209921",
        "5915211623",
        "5915716111",
        "5917058918",
        "5918743805",
        "5920993291",
        "5924430391",
        "5911183098",
        "5913253454",
        "5915261814",
        "5915724586",
        "5917079092",
        "5918832729",
        "5921084683",
        "5924507520",
        "5911258834",
        "5913299413",
        "5915269041",
        "5915726519",
        "5917120517",
        "5918906877",
        "5922162182",
        "5924610089"
    ]

max_tick = 0
X = []
y = []
new_X = []
for match_id in authentic_match_id:
# match_id = authentic_match_id[3]
    df_cursor = pd.read_csv("../data_collector/features/" + match_id + "_cursor.csv")
    df_match_info = pd.read_csv("../data_collector/features/" + match_id + "_info.csv")

    dfs = [rows for _, rows in df_cursor.groupby('Hero')]
    for cursor_info in dfs:
        steam_id = df_match_info.loc[df_match_info["Hero"] == cursor_info["Hero"].iloc[0]].iloc[0]["SteamId"]
        max_tick = cursor_info.shape[0] if cursor_info.shape[0] > max_tick else max_tick
        X.append(cursor_info.drop("Hero", axis=1))
        if steam_id == 76561198134243802:
            y.append(1)
        else:
            y.append(0)
for cursor_info in X:
    cursor_info["V_X"] = cursor_info["X"].diff() / cursor_info["Tick"].diff()
    cursor_info["V_Y"] = cursor_info["Y"].diff() / cursor_info["Tick"].diff()
    cursor_info["V"] = (cursor_info["V_X"]**2 + cursor_info["V_Y"]**2)**(1/2)
    cursor_info["A"] = cursor_info["V"].diff() / cursor_info["Tick"].diff()
    cursor_info["J"] = cursor_info["A"].diff() / cursor_info["Tick"].diff()
    cursor_info["AoM"] = np.arctan(cursor_info["X"].diff() / cursor_info["Y"].diff())
    cursor_info["AoM"] = cursor_info["AoM"].cumsum()
    cursor_info["Ang_V"] = cursor_info["AoM"].diff() / cursor_info["Tick"].diff()
    cursor_info.fillna(0, inplace=True)
    cursor_info.drop("Tick", axis=1, inplace=True)
    cursor_info.drop("X", axis=1, inplace=True)
    cursor_info.drop("Y", axis=1, inplace=True)
    cursor_info.replace([np.inf, -np.inf], 0, inplace=True)
    if cursor_info.shape[0] < max_tick:
        pddddd = pd.DataFrame({
                                # "Tick": np.zeros(max_tick-cursor_info.shape[0]),
                                # "X": np.zeros(max_tick-cursor_info.shape[0]), 
                                # "Y": np.zeros(max_tick-cursor_info.shape[0]), 
                                "V_X": np.zeros(max_tick-cursor_info.shape[0]), 
                                "V_Y": np.zeros(max_tick-cursor_info.shape[0]), 
                                "V": np.zeros(max_tick-cursor_info.shape[0]), 
                                "A": np.zeros(max_tick-cursor_info.shape[0]), 
                                "J": np.zeros(max_tick-cursor_info.shape[0]), 
                                "AoM": np.zeros(max_tick-cursor_info.shape[0]), 
                                "Ang_V": np.zeros(max_tick-cursor_info.shape[0])})

        cursor_info = cursor_info.append(pddddd, ignore_index = True)
    new_X.append(cursor_info.to_numpy().flatten())
    
X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.20, random_state=15)

# In[3]: Calculate eer rate
from sklearn.metrics import make_scorer, roc_curve #for eer
from scipy.optimize import brentq #for eer
from scipy.interpolate import interp1d #for eer
def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer



# In[4]: Logistic Regression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=15).fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('The accuracy of the Logistic Regression is ', round(accuracy_score(prediction_rm, y_test)*100,2))
print('The precision of the Logistic Regression is ', round(precision_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The recall of the Logistic Regression is ', round(recall_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The f1_score of the Logistic Regression is ', round(f1_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
# print('The EER value of the Logistic Regression is ', round(calculate_eer(prediction_rm, y_test)*100,2))



# In[5]: Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=15).fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('The accuracy of the Decision Tree is ', round(accuracy_score(prediction_rm, y_test)*100,2))
print('The precision of the Decision Tree is ', round(precision_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The recall of the Decision Tree is ', round(recall_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The f1_score of the Decision Tree is ', round(f1_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
# print('The EER value of the Decision Tree is ', round(calculate_eer(prediction_rm, y_test)*100,2))


# In[6]: Random Forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=15).fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('The accuracy of the Random Forest is ', round(accuracy_score(prediction_rm, y_test)*100,2))
print('The precision of the Random Forest is ', round(precision_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The recall of the Random Forest is ', round(recall_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
print('The f1_score of the Random Forest is ', round(f1_score(prediction_rm, y_test, pos_label=1, average='binary')*100,2))
# print('The EER value of the Random Forest is ', round(calculate_eer(prediction_rm, y_test)*100,2))


# %%
