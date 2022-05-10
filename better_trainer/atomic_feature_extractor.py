# In[1]: Headerimport

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #for accuracy_score and precision_score
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data

from sklearn.model_selection import cross_validate #score evaluation

# In[1]: Other stuff
df_cursor = pd.read_csv("cursor.csv")
df_cursor = df_cursor[df_cursor['Tick'] > 29999]
df_unit_order = pd.read_csv("unit_order.csv")
df_unit_order = df_unit_order[df_unit_order['Tick'] > 29999]
df_unit_order.drop_duplicates(inplace=True)
df_unit_order.replace({"Action": {'M': 1, 'A': 2, 'S': 3},}, inplace=True)
df_match_info = pd.read_csv("info.csv")
dfs_cursor = [rows for _, rows in df_cursor.groupby('Hero')]
dfs_unit_order = [rows for _, rows in df_unit_order.groupby('Hero')]
for hero in dfs_unit_order:
    steam_id = df_match_info.loc[df_match_info["Hero"] == hero["Hero"].iloc[0]].iloc[0]["SteamId"]
    hero_name = df_match_info.loc[df_match_info["Hero"] == hero["Hero"].iloc[0]].iloc[0]["Hero"]
    hero = hero.groupby(hero['Tick']).aggregate({'Action': 'max'}).reset_index()
    df_hero_cursor = list(filter(lambda x: x.iloc[0]["Hero"] == hero_name, dfs_cursor))[0]

    cols = ["X", "Y"]
    df_hero_tmp = df_hero_cursor.loc[(df_hero_cursor[cols].shift(-1) != df_hero_cursor[cols]).any(axis=1)]
    df_hero_tmp["seq"] = df_hero_tmp["Tick"].diff()
    tbd_index = pd.cut(df_hero_tmp[df_hero_tmp["seq"] > 10]["Tick"], hero["Tick"].values).drop_duplicates(keep="last").index.values[:-1]
    df_hero_cursor["Action"] = pd.cut(df_hero_cursor["Tick"], bins=hero["Tick"].values, labels=hero["Action"].iloc[1:].values, ordered=False)
    df_hero_cursor["Action"] = pd.to_numeric(df_hero_cursor["Action"])
    df_hero_cursor["range"] = pd.cut(df_hero_cursor["Tick"], hero["Tick"].values)
    for rm in tbd_index:
        df_hero_cursor.drop(df_hero_cursor[(df_hero_cursor.loc[rm]["range"].left <= df_hero_cursor["Tick"]) & (df_hero_cursor["Tick"] <= df_hero_cursor.loc[rm]["Tick"])].index, inplace=True)
    df_hero_cursor.dropna(inplace=True)

    df_hero_cursor["V_X"] = df_hero_cursor["X"].diff() / df_hero_cursor["Tick"].diff()
    df_hero_cursor["V_Y"] = df_hero_cursor["Y"].diff() / df_hero_cursor["Tick"].diff()
    df_hero_cursor["V"] = np.sqrt(df_hero_cursor["V_X"]**2 + df_hero_cursor["V_Y"]**2)
    df_hero_cursor["S"] = np.sqrt(df_hero_cursor["X"].diff()**2 + df_hero_cursor["Y"].diff()**2)
    # df_hero_cursor.fillna({"S":0}, inplace=True)
    # df_hero_cursor["S"] = np.cumsum(df_hero_cursor["S"])
    df_hero_cursor["A"] = df_hero_cursor["V"].diff() / df_hero_cursor["Tick"].diff()
    df_hero_cursor["A_X"] = df_hero_cursor["V_X"].diff() / df_hero_cursor["Tick"].diff()
    df_hero_cursor["A_Y"] = df_hero_cursor["V_Y"].diff() / df_hero_cursor["Tick"].diff()
    df_hero_cursor["J"] = df_hero_cursor["A"].diff() / df_hero_cursor["Tick"].diff()
    df_hero_cursor["AoM"] = np.arctan2(df_hero_cursor["Y"], df_hero_cursor["X"])
    df_hero_cursor["AoM"] = df_hero_cursor["AoM"].diff()
    # df_hero_cursor.fillna({"AoM":0}, inplace=True)
    # df_hero_cursor["AoM"] = np.cumsum(df_hero_cursor["AoM"])
    df_hero_cursor["Ang_V"] = df_hero_cursor["AoM"] / df_hero_cursor["Tick"].diff()
    df_hero_cursor["Cur"] = df_hero_cursor["AoM"] / df_hero_cursor["S"]
    df_hero_cursor["Cur_cr"] = df_hero_cursor["Cur"].diff() / df_hero_cursor["S"]
    df_hero_cursor["Y_diff"] = df_hero_cursor["Y"].diff()
    df_hero_cursor["X_diff"] = df_hero_cursor["X"].diff()
    df_hero_cursor["TCM"] = df_hero_cursor["Tick"] * np.sqrt(df_hero_cursor["X_diff"] ** 2 + df_hero_cursor["Y_diff"] ** 2)
    df_hero_cursor["SC"] = (df_hero_cursor["Tick"].diff() ** 2) * np.sqrt(df_hero_cursor["X_diff"] ** 2 + df_hero_cursor["Y_diff"] ** 2)
    df_hero_cursor["M3"] = (df_hero_cursor["Tick"].diff() ** 3) * np.sqrt(df_hero_cursor["X_diff"] ** 2 + df_hero_cursor["Y_diff"] ** 2)
    df_hero_cursor["M4"] = (df_hero_cursor["Tick"].diff() ** 4) * np.sqrt(df_hero_cursor["X_diff"] ** 2 + df_hero_cursor["Y_diff"] ** 2)
    df_hero_cursor["TCrv"] =    (df_hero_cursor["V_X"] * df_hero_cursor["A_Y"] 
                                - df_hero_cursor["V_Y"] * df_hero_cursor["A_X"])\
                                / np.power(df_hero_cursor["V_X"] ** 2 + df_hero_cursor["V_Y"] ** 2, 1.5)
    df_hero_cursor["VCrv"] = df_hero_cursor["J"] / np.power(1 + df_hero_cursor["A"] ** 2, 1.5)
    print("bruh")
    df_hero_cursor.fillna({"V_X":0, "V_Y":0, "V":0, "A":0, "J":0, "AoM":0, "Ang_V":0, "Cur":0, "Cur_cr":0, "Y_diff":0, "X_diff":0}, inplace=True)
    atomic_order = df_hero_cursor.groupby("range").agg(
        Tick=("Tick", "count"), 
        distance=("S", "sum"), 
        Action=("Action", "first"),
        X_first=("X", "first"),
        X_last=("X", "last"),
        Y_first=("Y", "first"),
        Y_last=("Y", "last"),
        X_diff=("X_diff", "sum"),
        Y_diff=("Y_diff", "sum"),
        V_X_min=("V_X", "min"),
        V_X_max=("V_X", "max"),
        V_X_mean=("V_X", "mean"),
        V_X_std=("V_X", "std"),
        V_Y_min=("V_Y", "min"),
        V_Y_max=("V_Y", "max"),
        V_Y_mean=("V_Y", "mean"),
        V_Y_std=("V_Y", "std"),
        V_min=("V", "min"),
        V_max=("V", "max"),
        V_mean=("V", "mean"),
        V_std=("V", "std"),
        A_min=("A", "min"),
        A_max=("A", "max"),
        A_mean=("A", "mean"),
        A_std=("A", "std"),
        J_min=("J", "min"),
        J_max=("J", "max"),
        J_mean=("J", "mean"),
        J_std=("J", "std"),
        AoM_min=("AoM", "min"),
        AoM_max=("AoM", "max"),
        AoM_mean=("AoM", "mean"),
        AoM_std=("AoM", "std"),
        Ang_V_min=("Ang_V", "min"),
        Ang_V_max=("Ang_V", "max"),
        Ang_V_mean=("Ang_V", "mean"),
        Ang_V_std=("Ang_V", "std"),
        Cur_min=("Cur", "min"),
        Cur_max=("Cur", "max"),
        Cur_mean=("Cur", "mean"),
        Cur_std=("Cur", "std"),
        TCM = ("TCM", "sum"),
        SC = ("SC", "sum"),
        M3 = ("M3", "sum"),
        M4 = ("M4", "sum"),
        TCrv = ("TCrv", "mean"),
        VCrv = ("VCrv", "mean"),
        Cur_cr_min=("Cur_cr", "min"),
        Cur_cr_max=("Cur_cr", "max"),
        Cur_cr_mean=("Cur_cr", "mean"),
        Cur_cr_std=("Cur_cr", "std")
        ).fillna(0).replace([np.inf, -np.inf], 0)
    atomic_order["S"] = np.sqrt((atomic_order["X_first"] - atomic_order["X_last"]) ** 2 + (atomic_order["Y_first"] - atomic_order["Y_last"]) ** 2) / atomic_order["distance"]
    atomic_order["TCM"] = atomic_order["TCM"] / atomic_order["distance"]
    atomic_order["SC"] = atomic_order["SC"] / atomic_order["distance"] - atomic_order["TCM"] ** 2
    atomic_order["M3"] = atomic_order["M3"] / atomic_order["distance"]
    atomic_order["M4"] = atomic_order["M4"] / atomic_order["distance"]
    atomic_order.drop("X_diff", axis=1 ,inplace=True)
    atomic_order.drop("Y_diff", axis=1 ,inplace=True)
    atomic_order.drop(atomic_order[atomic_order["Tick"] < 8].index, inplace = True)
    atomic_order["Steam_id"] = steam_id
    atomic_order["Hero"] = hero_name
    df_hero_cursor.drop("S", axis=0, inplace=True)
    


# In[2]: Complex mouse action
replay_tracker = pd.read_csv("replay_tracker.csv")

authentic_match_id = replay_tracker.loc[(replay_tracker['state'] == 6) & (replay_tracker['1_tick'] == True), 'replay_id'].tolist()

counter = 1

not_yet = True

for match_id in authentic_match_id:
    print(match_id)
    match_id = str(match_id)
    if match_id != "6301239781":
        if not_yet:
            counter += 1
            continue
    else:
        not_yet = False
        print("lets go")
        counter += 1
        continue
    df_cursor = pd.read_csv("../automation/features/" + match_id + "_cursor_tmp_1_tick.csv")
    df_cursor = df_cursor[df_cursor['Tick'] > 29999]
    df_unit_order = pd.read_csv("../automation/features/"  + match_id + "_unit_order_v2.csv")
    df_unit_order = df_unit_order[df_unit_order['Tick'] > 29999]
    df_unit_order.drop_duplicates(inplace=True)
    df_unit_order.replace({"Action": {'M': 1, 'A': 2, 'S': 3},}, inplace=True)
    df_match_info = pd.read_csv("../automation/features/"  + match_id + "_info.csv")

    dfs_cursor = [rows for _, rows in df_cursor.groupby('Hero')]
    dfs_unit_order = [rows for _, rows in df_unit_order.groupby('Hero')]
    for hero in dfs_unit_order:
        steam_id = df_match_info.loc[df_match_info["Hero"] == hero["Hero"].iloc[0]].iloc[0]["SteamId"]
        hero_name = df_match_info.loc[df_match_info["Hero"] == hero["Hero"].iloc[0]].iloc[0]["Hero"]
        hero = hero.groupby(hero['Tick']).aggregate({'Action': 'max'}).reset_index()
        try:
            df_hero_cursor = list(filter(lambda x: x.iloc[0]["Hero"] == hero_name, dfs_cursor))[0]
        except:
            continue

        cols = ["X", "Y"]
        df_hero_tmp = df_hero_cursor.loc[(df_hero_cursor[cols].shift(-1) != df_hero_cursor[cols]).any(axis=1)]
        df_hero_tmp["seq"] = df_hero_tmp["Tick"].diff()
        tbd_index = pd.cut(df_hero_tmp[df_hero_tmp["seq"] > 10]["Tick"], hero["Tick"].values).drop_duplicates(keep="last").index.values[:-1]
        df_hero_cursor["Action"] = pd.cut(df_hero_cursor["Tick"], bins=hero["Tick"].values, labels=hero["Action"].iloc[1:].values, ordered=False)
        df_hero_cursor["Action"] = pd.to_numeric(df_hero_cursor["Action"])
        df_hero_cursor["range"] = pd.cut(df_hero_cursor["Tick"], hero["Tick"].values)
        for rm in tbd_index:
            df_hero_cursor.drop(df_hero_cursor[(df_hero_cursor.loc[rm]["range"].left <= df_hero_cursor["Tick"]) & (df_hero_cursor["Tick"] <= df_hero_cursor.loc[rm]["Tick"])].index, inplace=True)
        
        df_hero_cursor.dropna(inplace=True)
        df_hero_cursor["V_X"] = df_hero_cursor["X"].diff() / df_hero_cursor["Tick"].diff()
        df_hero_cursor["V_Y"] = df_hero_cursor["Y"].diff() / df_hero_cursor["Tick"].diff()
        df_hero_cursor["V"] = np.sqrt(df_hero_cursor["V_X"]**2 + df_hero_cursor["V_Y"]**2)
        df_hero_cursor["S"] = np.sqrt(df_hero_cursor["X"].diff()**2 + df_hero_cursor["Y"].diff()**2)
        # df_hero_cursor.fillna({"S":0}, inplace=True)
        # df_hero_cursor["S"] = np.cumsum(df_hero_cursor["S"])
        df_hero_cursor["A_X"] = df_hero_cursor["V_X"].diff() / df_hero_cursor["Tick"].diff()
        df_hero_cursor["A_Y"] = df_hero_cursor["V_Y"].diff() / df_hero_cursor["Tick"].diff()
        df_hero_cursor["A"] = df_hero_cursor["V"].diff() / df_hero_cursor["Tick"].diff()
        df_hero_cursor["J"] = df_hero_cursor["A"].diff() / df_hero_cursor["Tick"].diff()
        df_hero_cursor["AoM"] = np.arctan2(df_hero_cursor["Y"], df_hero_cursor["X"])
        df_hero_cursor["AoM"] = df_hero_cursor["AoM"].diff()
        # df_hero_cursor.fillna({"AoM":0}, inplace=True)
        # df_hero_cursor["AoM"] = np.cumsum(df_hero_cursor["AoM"])
        df_hero_cursor["Ang_V"] = df_hero_cursor["AoM"] / df_hero_cursor["Tick"].diff()
        df_hero_cursor["Cur"] = df_hero_cursor["AoM"] / df_hero_cursor["S"]
        df_hero_cursor["Cur_cr"] = df_hero_cursor["Cur"].diff() / df_hero_cursor["S"]
        df_hero_cursor["Y_diff"] = df_hero_cursor["Y"].diff()
        df_hero_cursor["X_diff"] = df_hero_cursor["X"].diff()
        df_hero_cursor["TCM"] = df_hero_cursor["Tick"] * np.sqrt(df_hero_cursor["X_diff"] ** 2 + df_hero_cursor["Y_diff"] ** 2)
        df_hero_cursor["SC"] = (df_hero_cursor["Tick"] ** 2) * np.sqrt(df_hero_cursor["X_diff"] ** 2 + df_hero_cursor["Y_diff"] ** 2)
        df_hero_cursor["M3"] = (df_hero_cursor["Tick"] ** 3) * np.sqrt(df_hero_cursor["X_diff"] ** 2 + df_hero_cursor["Y_diff"] ** 2)
        df_hero_cursor["M4"] = (df_hero_cursor["Tick"] ** 4) * np.sqrt(df_hero_cursor["X_diff"] ** 2 + df_hero_cursor["Y_diff"] ** 2)
        df_hero_cursor["TCrv"] = (df_hero_cursor["V_X"] * df_hero_cursor["A_Y"] \
                                - df_hero_cursor["V_Y"] * df_hero_cursor["A_X"])\
                                / np.power(df_hero_cursor["V_X"] ** 2 + df_hero_cursor["V_Y"] ** 2, 1.5)
        df_hero_cursor["VCrv"] = df_hero_cursor["J"] / np.power(1 + df_hero_cursor["A"] ** 2, 1.5)
        df_hero_cursor.fillna({"V_X":0, "V_Y":0, "V":0, "A":0, "J":0, "AoM":0, "Ang_V":0, "Cur":0, "Cur_cr":0, "Y_diff":0, "X_diff":0}, inplace=True)
        atomic_order = df_hero_cursor.groupby("range").agg(
            Tick=("Tick", "count"), 
            distance=("S", "sum"), 
            Action=("Action", "first"),
            X_first=("X", "first"),
            X_last=("X", "last"),
            Y_first=("Y", "first"),
            Y_last=("Y", "last"),
            X_diff=("X_diff", "sum"),
            Y_diff=("Y_diff", "sum"),
            V_X_min=("V_X", "min"),
            V_X_max=("V_X", "max"),
            V_X_mean=("V_X", "mean"),
            V_X_std=("V_X", "std"),
            V_Y_min=("V_Y", "min"),
            V_Y_max=("V_Y", "max"),
            V_Y_mean=("V_Y", "mean"),
            V_Y_std=("V_Y", "std"),
            V_min=("V", "min"),
            V_max=("V", "max"),
            V_mean=("V", "mean"),
            V_std=("V", "std"),
            A_min=("A", "min"),
            A_max=("A", "max"),
            A_mean=("A", "mean"),
            A_std=("A", "std"),
            J_min=("J", "min"),
            J_max=("J", "max"),
            J_mean=("J", "mean"),
            J_std=("J", "std"),
            AoM_min=("AoM", "min"),
            AoM_max=("AoM", "max"),
            AoM_mean=("AoM", "mean"),
            AoM_std=("AoM", "std"),
            Ang_V_min=("Ang_V", "min"),
            Ang_V_max=("Ang_V", "max"),
            Ang_V_mean=("Ang_V", "mean"),
            Ang_V_std=("Ang_V", "std"),
            Cur_min=("Cur", "min"),
            Cur_max=("Cur", "max"),
            Cur_mean=("Cur", "mean"),
            Cur_std=("Cur", "std"),
            TCM = ("TCM", "sum"),
            SC = ("SC", "sum"),
            M3 = ("M3", "sum"),
            M4 = ("M4", "sum"),
            TCrv = ("TCrv", "mean"),
            VCrv = ("VCrv", "mean"),
            Cur_cr_min=("Cur_cr", "min"),
            Cur_cr_max=("Cur_cr", "max"),
            Cur_cr_mean=("Cur_cr", "mean"),
            Cur_cr_std=("Cur_cr", "std")
            ).fillna(0).replace([np.inf, -np.inf], 0)
        
        atomic_order["S"] = np.sqrt((atomic_order["X_first"] - atomic_order["X_last"]) ** 2 + (atomic_order["Y_first"] - atomic_order["Y_last"]) ** 2) / atomic_order["distance"]
        atomic_order["TCM"] = atomic_order["TCM"] / atomic_order["distance"]
        atomic_order["SC"] = atomic_order["SC"] / atomic_order["distance"] - atomic_order["TCM"] ** 2
        atomic_order["M3"] = atomic_order["M3"] / atomic_order["distance"]
        atomic_order["M4"] = atomic_order["M4"] / atomic_order["distance"]
        
        atomic_order.drop("X_diff", axis=1 ,inplace=True)
        atomic_order.drop("Y_diff", axis=1 ,inplace=True)
        atomic_order.drop(atomic_order[atomic_order["Tick"] < 8].index, inplace = True)
        atomic_order["Steam_id"] = steam_id
        atomic_order["Hero"] = hero_name
        # df_hero_cursor.drop("S", axis=0, inplace=True)
        if (atomic_order[atomic_order["Action"] == 1].index.empty or atomic_order[atomic_order["Action"] == 2].index.empty or atomic_order[atomic_order["Action"] == 3].index.empty):
                 print("Empty action")
                 break
        atomic_order_move = atomic_order.drop(atomic_order[atomic_order["Action"] != 1].index)
        atomic_order_attack = atomic_order.drop(atomic_order[atomic_order["Action"] != 2].index)
        atomic_order_spell = atomic_order.drop(atomic_order[atomic_order["Action"] != 3].index)
        atomic_order.to_csv("trainable_features/combo/" + str(steam_id) + "|" + hero_name + ".csv", index=False)
        atomic_order_move.to_csv("trainable_features/move/" + str(steam_id) + "|" + hero_name + "|move.csv", index=False)
        atomic_order_attack.to_csv("trainable_features/attack/" + str(steam_id) + "|" + hero_name + "|attack.csv", index=False)
        atomic_order_spell.to_csv("trainable_features/spell/" + str(steam_id) + "|" + hero_name + "|spell.csv", index=False)


    print("batch " + str(counter) + "/" + str(len(authentic_match_id)))
    counter += 1


# %%
