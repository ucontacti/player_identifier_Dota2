import requests
import json
import sys
import time
import datetime

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

if __name__ == "__main__":
    hero, replay_list = list_replay_by_match_id(sys.argv[1])    
    print(get_steam64_from_steam32(sys.argv[1]))