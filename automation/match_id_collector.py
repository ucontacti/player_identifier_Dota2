import requests
import json
import sys
import time
import datetime

if __name__ == "__main__":
    d = datetime.datetime.now() - datetime.timedelta(days=14)
    unixtime = time.mktime(d.timetuple())

    response = requests.get("https://api.opendota.com/api/players/" + sys.argv[1] + "/matches")

    for i in response.json():
        if i["hero_id"] == 102 and i["start_time"] > unixtime:
            print(i["match_id"])