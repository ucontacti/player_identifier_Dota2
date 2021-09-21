import requests
import sys
import time
import datetime
import os.path
import pandas as pd
from replay_downloader import replay_download

REPLAY_TRACKER_PATH = "automation/replay_tracker.csv"
PLAYER_ID_PATH = "automation/player_id.txt"


def list_replay_by_match_id(player_id):
    d = datetime.datetime.now() - datetime.timedelta(days=14)
    unixtime = time.mktime(d.timetuple())

    response = requests.get("https://api.opendota.com/api/players/" + player_id + "/matches")
    hero_dic = {}
    for i in response.json():
        if i["start_time"] > unixtime:
            if i["hero_id"] not in hero_dic:
                hero_dic[i["hero_id"]] = 1
            else:
                hero_dic[i["hero_id"]] += 1
    hero_dic = sorted(hero_dic.items(), key=lambda item: item[1], reverse=True)
    max_hero_replay = hero_dic[0][0]
    no_max_replay = hero_dic[0][1]
    if no_max_replay >= 40:
        replay_list = []
        for i in response.json():
            if i["start_time"] > unixtime and i["hero_id"] == max_hero_replay:
                replay_list.append(i["match_id"])
        return max_hero_replay, replay_list
    else:
        return 0, []

def get_steam64_from_steam32(steam32):
    return int(steam32) + 76561197960265728

def add_to_replay_tracker():
    replay_tracker = pd.read_csv(REPLAY_TRACKER_PATH)
    players_id = open(PLAYER_ID_PATH, "r").read().split("\n")
    new_replays = []
    for player in players_id:
        hero, replay_list = list_replay_by_match_id(player)
        for replay in replay_list:
            if replay not in replay_tracker["replay_id"].values:
                new_dict = {
                    'player_id': player, 
                    'player_64_id': get_steam64_from_steam32(player), 
                    'replay_id': replay, 
                    'hero': hero, 
                    'state': 0,
                    'click_rate': 0
                }
                new_replays.append(new_dict)
    new_plays_df = pd.DataFrame(new_replays)
    pd.concat([replay_tracker, new_plays_df]).to_csv(REPLAY_TRACKER_PATH, index=False)

def create_empty_replay_tracker():
    pd.DataFrame(columns=['player_id', 'player_64_id', 'replay_id', 'hero', 'state','click_rate']).to_csv(REPLAY_TRACKER_PATH, index=False)

def update_replay_tracker():
    replay_tracker = pd.read_csv(REPLAY_TRACKER_PATH)
    new_val = replay_download(replay_tracker.loc[(replay_tracker['state'] == 0), 'replay_id'].tolist())
    replay_tracker.loc[(replay_tracker['state'] == 0),'state'] = new_val
    new_val = feature_extractor(replay_tracker.loc[(replay_tracker['state'] == 1), 'replay_id'].tolist())
    replay_tracker.to_csv(REPLAY_TRACKER_PATH, index=False)

if __name__ == "__main__":
    # hero, replay_list = list_replay_by_match_id(sys.argv[1])    
    # print(get_steam64_from_steam32(sys.argv[1]))
    if os.path.isfile(REPLAY_TRACKER_PATH):
        # add_to_replay_tracker()
        update_replay_tracker()
    else:
        create_empty_replay_tracker()
        # add_to_replay_tracker()
        update_replay_tracker()