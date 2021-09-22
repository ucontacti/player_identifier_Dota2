import numpy as np 
import pandas as pd

def atomic_feature(match_id_list, tick_rate):
    new_X = []
    counter = 1
    new_X_mov = []
    new_X_att = []
    new_X_spl = []
    batch_number = 1
    new_val = []



    for match_id in match_id_list:
        df_cursor = pd.read_csv("../data_collector/features/" + match_id + "_cursor_tmp_"  + tick_rate + "_tick.csv")
        df_unit_order = pd.read_csv("../data_collector/features/" + match_id + "_unit_order_v2.csv")
        df_unit_order.drop_duplicates(inplace=True)
        df_unit_order.replace({"Action": {'M': 1, 'A': 2, 'S': 3},}, inplace=True)
        df_match_info = pd.read_csv("../data_collector/features/" + match_id + "_info.csv")

        dfs_cursor = [rows for _, rows in df_cursor.groupby('Hero')]
        dfs_unit_order = [rows for _, rows in df_unit_order.groupby('Hero')]
        for hero in dfs_unit_order:
            steam_id = df_match_info.loc[df_match_info["Hero"] == hero["Hero"].iloc[0]].iloc[0]["SteamId"]
            hero_name = df_match_info.loc[df_match_info["Hero"] == hero["Hero"].iloc[0]].iloc[0]["Hero"]
            hero = hero.groupby(hero['Tick']).aggregate({'Action': 'max'}).reset_index()
            df_hero_cursor = list(filter(lambda x: x.iloc[0]["Hero"] == hero_name, dfs_cursor))[0]
            df_hero_cursor["Action"] = pd.cut(df_hero_cursor["Tick"], bins=hero["Tick"].values, labels=hero["Action"].iloc[1:].values, ordered=False)
            df_hero_cursor["range"] = pd.cut(df_hero_cursor["Tick"], hero["Tick"].values)
            df_hero_cursor.dropna(inplace=True)
            df_hero_cursor["V_X"] = df_hero_cursor["X"].diff() / df_hero_cursor["Tick"].diff()
            df_hero_cursor["V_Y"] = df_hero_cursor["Y"].diff() / df_hero_cursor["Tick"].diff()
            df_hero_cursor["V"] = np.sqrt(df_hero_cursor["V_X"]**2 + df_hero_cursor["V_Y"]**2)
            df_hero_cursor["S"] = np.sqrt((df_hero_cursor["X"].diff())**2 + (df_hero_cursor["Y"].diff())**2)
            df_hero_cursor.fillna({"S":0}, inplace=True)
            df_hero_cursor["S"] = np.cumsum(df_hero_cursor["S"])
            df_hero_cursor["A"] = df_hero_cursor["V"].diff() / df_hero_cursor["Tick"].diff()
            df_hero_cursor["J"] = df_hero_cursor["A"].diff() / df_hero_cursor["Tick"].diff()
            df_hero_cursor["AoM"] = np.arctan(df_hero_cursor["Y"].diff() / df_hero_cursor["X"].diff())
            df_hero_cursor.fillna({"AoM":0}, inplace=True)
            df_hero_cursor["AoM"] = np.cumsum(df_hero_cursor["AoM"])
            df_hero_cursor["Ang_V"] = df_hero_cursor["AoM"].diff() / df_hero_cursor["Tick"].diff()
            df_hero_cursor["Cur"] = df_hero_cursor["AoM"].diff() / df_hero_cursor["S"].diff()
            df_hero_cursor["Y_diff"] = df_hero_cursor["Y"].diff()
            df_hero_cursor["X_diff"] = df_hero_cursor["X"].diff()
            # df_hero_cursor.drop("S", axis=0, inplace=True)
            # df_hero_cursor["Cur_cr"] = df_hero_cursor["Cur"].diff() / df_hero_cursor["S"].diff()
            df_hero_cursor.fillna({"V_X":0, "V_Y":0, "V":0, "A":0, "J":0, "AoM":0, "Ang_V":0, "Cur":0, "Y_diff":0, "X_diff":0}, inplace=True)
            atomic_order = df_hero_cursor.groupby("range").agg({
                "Tick": "count", 
                "Action": "first",
                "X_diff": "sum",
                "Y_diff": "sum",
                "V_X":["min", "max", "mean", "std"], 
                "V_Y":["min", "max", "mean", "std"],
                "V":["min", "max", "mean", "std"],
                "A":["min", "max", "mean", "std"],
                "J":["min", "max", "mean", "std"],
                "AoM":["min", "max", "mean", "std"],
                "Ang_V":["min", "max", "mean", "std"],
                "Cur":["min", "max", "mean", "std"]
                # "Cur_cr":["min", "max", "mean", "std"],
                }).fillna(0).replace([np.inf, -np.inf], 0)
            atomic_order["distance"] = np.sqrt(atomic_order["X_diff"]["sum"]**2 + atomic_order["Y_diff"]["sum"]**2)
            atomic_order.drop("X_diff", axis=1 ,inplace=True)
            atomic_order.drop("Y_diff", axis=1 ,inplace=True)
            atomic_order.drop(atomic_order[atomic_order["Tick"]["count"] < 2].index, inplace = True)
            atomic_order_move = atomic_order.drop(atomic_order[atomic_order["Action"]["first"] != 1].index)
            atomic_order_attack = atomic_order.drop(atomic_order[atomic_order["Action"]["first"] != 2].index)
            atomic_order_spell = atomic_order.drop(atomic_order[atomic_order["Action"]["first"] != 3].index)
            atomic_order_move_arr = [[steam_id, hero_name], atomic_order_move.to_numpy().flatten()]
            atomic_order_attack_arr = [[steam_id, hero_name], atomic_order_attack.to_numpy().flatten()]
            atomic_order_spell_arr = [[steam_id, hero_name], atomic_order_spell.to_numpy().flatten()]
            atomic_order_arr = [[steam_id, hero_name], atomic_order.to_numpy().flatten()]
            new_X_mov.append(atomic_order_move_arr)
            new_X_att.append(atomic_order_attack_arr)
            new_X_spl.append(atomic_order_spell_arr)
            new_X.append(atomic_order_arr)
            new_val.append(6)

        print("batch " + str(counter) + "/" + str(len(match_id_list)))
        counter += 1
        if (counter%100) == 0:
            filename = "atomic_move_v5_tickrate_" + tick_rate + "_batch_" + batch_number + ".npy"
            np.save(filename, new_X_mov, allow_pickle=True)
            filename = "atomic_attack_v5_tickrate_" + tick_rate + "_batch_" + batch_number + ".npy"
            np.save(filename, new_X_att, allow_pickle=True)
            filename = "atomic_spell_v5_tickrate_" + tick_rate + "_batch_" + batch_number + ".npy"
            np.save(filename, new_X_spl, allow_pickle=True)
            filename = "atomic_v5_tickrate_" + tick_rate + "_batch_" + batch_number + ".npy"
            np.save(filename, new_X, allow_pickle=True)
            new_X = []
            new_X_mov = []
            new_X_att = []
            new_X_spl = []
            batch_number += 1
            counter = 1
    if new_X:
        filename = "atomic_move_v5_tickrate_" + tick_rate + "_batch_" + batch_number + ".npy"
        np.save(filename, new_X_mov, allow_pickle=True)
        filename = "atomic_attack_v5_tickrate_" + tick_rate + "_batch_" + batch_number + ".npy"
        np.save(filename, new_X_att, allow_pickle=True)
        filename = "atomic_spell_v5_tickrate_" + tick_rate + "_batch_" + batch_number + ".npy"
        np.save(filename, new_X_spl, allow_pickle=True)
        filename = "atomic_v5_tickrate_" + tick_rate + "_batch_" + batch_number + ".npy"
        np.save(filename, new_X, allow_pickle=True)
    return new_val

