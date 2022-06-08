import requests
import time
import datetime
from os.path import exists
import pandas as pd
from util.replay_downloader import replay_download
from util.feature_builder import replay_decompress, game_info,        \
                   unit_order, item_change, item_all,   \
                   cursor_data
from util.atomic_feature_extractor import atomic_feature
REPLAY_TRACKER_PATH = "../resources/replay_tracker.csv"
PLAYER_ID_PATH = "../resources/player_index.txt"


def list_replay_by_match_id(player_id):
    d = datetime.datetime.now() - datetime.timedelta(days=10)
    unixtime = time.mktime(d.timetuple())

    response = requests.get("https://api.opendota.com/api/players/" + player_id + "/matches")
    hero_dic = {}
    for i in response.json():
        if i == "error":
            continue
        if i["start_time"] > unixtime:
            if i["hero_id"] not in hero_dic:
                hero_dic[i["hero_id"]] = 1
            else:
                hero_dic[i["hero_id"]] += 1
    print(player_id)
    if not bool(hero_dic):
        return 0, []
    hero_dic = sorted(hero_dic.items(), key=lambda item: item[1], reverse=True)
    max_hero_replay = hero_dic[0][0]
    no_max_replay = hero_dic[0][1]
    if no_max_replay >= 30:
        replay_list = []
        for i in response.json():
            if i["start_time"] > unixtime and i["hero_id"] == max_hero_replay:
                replay_list.append(i["match_id"])
        return max_hero_replay, replay_list
    else:
        return 0, []


def add_replay_to_pipeline():
    get_steam64_from_steam32 = lambda steam32: int(steam32) + 76561197960265728
    replay_tracker = pd.read_csv(REPLAY_TRACKER_PATH)
    players_id = open(PLAYER_ID_PATH, "r").read().split("\n")
    new_replays = []
    for counter, player in enumerate(players_id):
        hero, replay_list = list_replay_by_match_id(player)
        for replay in replay_list:
            if replay not in replay_tracker["replay_id"].values:
                new_dict = {
                    'player_id': player, 
                    'player_64_id': get_steam64_from_steam32(player), 
                    'replay_id': replay, 
                    'hero': hero, 
                    'state': 0,
                    '1_tick': False,
                    '5_tick': False,
                    '10_tick': False,
                    '15_tick': False,
                    '30_tick': False,
                }
                new_replays.append(new_dict)
        if (counter%58) == 0:
            print("sleeping and saving batch")
            time.sleep(60)
            new_plays_df = pd.DataFrame(new_replays)
            pd.concat([replay_tracker, new_plays_df]).to_csv(REPLAY_TRACKER_PATH, index=False)

        
    new_plays_df = pd.DataFrame(new_replays)
    pd.concat([replay_tracker, new_plays_df]).to_csv(REPLAY_TRACKER_PATH, index=False)

def create_empty_replay_tracker():
    pd.DataFrame(columns=['player_id', 'player_64_id', 'replay_id', 'hero', 'state', '1_tick', '5_tick', '10_tick', '15_tick', '30_tick']).to_csv(REPLAY_TRACKER_PATH, index=False)

def update_pipeline():
    replay_tracker = pd.read_csv(REPLAY_TRACKER_PATH)
    
    new_val = replay_download(replay_tracker.loc[(replay_tracker['state'] == 0), 'replay_id'].tolist())
    replay_tracker.loc[(replay_tracker['state'] == 0),'state'] = new_val
    replay_tracker.to_csv(REPLAY_TRACKER_PATH, index=False)

    new_val = replay_decompress(replay_tracker.loc[(replay_tracker['state'] == 1), 'replay_id'].tolist())
    replay_tracker.loc[(replay_tracker['state'] == 1),'state'] = new_val
    replay_tracker.to_csv(REPLAY_TRACKER_PATH, index=False)
    
    new_val = game_info(replay_tracker.loc[(replay_tracker['state'] == 2), 'replay_id'].tolist())
    replay_tracker.loc[(replay_tracker['state'] == 2),'state'] = new_val
    replay_tracker.to_csv(REPLAY_TRACKER_PATH, index=False)

    new_val = unit_order(replay_tracker.loc[(replay_tracker['state'] == 3), 'replay_id'].tolist())
    replay_tracker.loc[(replay_tracker['state'] == 3),'state'] = new_val
    replay_tracker.to_csv(REPLAY_TRACKER_PATH, index=False)

    new_val = item_change(replay_tracker.loc[(replay_tracker['state'] == 4), 'replay_id'].tolist())
    replay_tracker.loc[(replay_tracker['state'] == 4),'state'] = new_val
    replay_tracker.to_csv(REPLAY_TRACKER_PATH, index=False)

    new_val = item_all(replay_tracker.loc[(replay_tracker['state'] == 5), 'replay_id'].tolist())
    replay_tracker.loc[(replay_tracker['state'] == 5),'state'] = new_val
    replay_tracker.to_csv(REPLAY_TRACKER_PATH, index=False)

    new_val, tickrate = cursor_data(replay_tracker.loc[(replay_tracker['state'] == 6) & (replay_tracker['1_tick'] == False), 'replay_id'].tolist(), 1)
    if (new_val):
       replay_tracker.loc[(replay_tracker['state'] == 6),'state'] = new_val
       tickrate_column = str(tickrate) + "_tick"
       replay_tracker.loc[(replay_tracker['state'] == 7),tickrate_column] = True
       replay_tracker.to_csv(REPLAY_TRACKER_PATH, index=False)
    
    
    # new_val, tickrate = cursor_data(replay_tracker.loc[(replay_tracker['state'] == 7) & (replay_tracker['5_tick'] == False), 'replay_id'].tolist(), 5)
    # if (new_val):
    #    replay_tracker.loc[(replay_tracker['state'] == 7),'state'] = new_val
    #    tickrate_column = str(tickrate) + "_tick"
    #    replay_tracker.loc[(replay_tracker['state'] == 7),tickrate_column] = True
    #    replay_tracker.to_csv(REPLAY_TRACKER_PATH, index=False)
    # new_val, tickrate = cursor_data(replay_tracker.loc[(replay_tracker['state'] == 7) & (replay_tracker['10_tick'] == False), 'replay_id'].tolist(), 10)
    # if (new_val):
    #    replay_tracker.loc[(replay_tracker['state'] == 7),'state'] = new_val
    #    tickrate_column = str(tickrate) + "_tick"
    #    replay_tracker.loc[(replay_tracker['state'] == 7),tickrate_column] = True
    #    replay_tracker.to_csv(REPLAY_TRACKER_PATH, index=False)
    # new_val, tickrate = cursor_data(replay_tracker.loc[(replay_tracker['state'] == 7) & (replay_tracker['15_tick'] == False), 'replay_id'].tolist(), 15)
    # if (new_val):
    #    replay_tracker.loc[(replay_tracker['state'] == 7),'state'] = new_val
    #    tickrate_column = str(tickrate) + "_tick"
    #    replay_tracker.loc[(replay_tracker['state'] == 7),tickrate_column] = True
    #    replay_tracker.to_csv(REPLAY_TRACKER_PATH, index=False)
    # new_val, tickrate = cursor_data(replay_tracker.loc[(replay_tracker['state'] == 7) & (replay_tracker['30_tick'] == False), 'replay_id'].tolist(), 30)
    # if (new_val):
    #    replay_tracker.loc[(replay_tracker['state'] == 7),'state'] = new_val
    #    tickrate_column = str(tickrate) + "_tick"
    #    replay_tracker.loc[(replay_tracker['state'] == 7),tickrate_column] = True
    #    replay_tracker.to_csv(REPLAY_TRACKER_PATH, index=False)

    new_val = atomic_feature(replay_tracker.loc[(replay_tracker['state'] == 7), 'replay_id'].tolist())
    replay_tracker.loc[(replay_tracker['state'] == 7),'state'] = new_val
    
    replay_tracker.to_csv(REPLAY_TRACKER_PATH, index=False)