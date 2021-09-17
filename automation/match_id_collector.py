import requests
import json
import sys

if __name__ == "__main__":
    response = requests.get("https://api.opendota.com/api/players/" + sys.argv[1] + "/matches")
    for i in response.json():
        if i["hero_id"] == 102 and i["start_time"] > 1630673059:
            print(i["match_id"])