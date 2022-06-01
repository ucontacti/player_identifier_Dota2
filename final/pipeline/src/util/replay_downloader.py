import requests, json
import urllib
# TODO: Add submit request


def submit_parse_request(match):
    pass

def replay_download(match_id_list):
    new_value = []
    for counter, match in enumerate(match_id_list):
        match = str(match)
        stringurl = f"https://api.opendota.com/api/replays?match_id={match}"
        raw_data = requests.get(stringurl, allow_redirects=True)
        json_data = json.loads(raw_data.content.decode())
        
        try:
            demo_url = f"http://replay{json_data[0]['cluster']}.valve.net/570/{json_data[0]['match_id']}_{json_data[0]['replay_salt']}.dem.bz2"
        except Exception as e:
            new_value.append(-2)
            print(f"Unable to download replay. Error: {e}")
            continue        
        save_to = f"../resources/downloaded_replays/{match}.dem.bz2"
        print(f"Getting ready to download replay {counter} match_id: {match}")
        try:
            urllib.request.urlretrieve(demo_url, save_to)
        except Exception as e:
            new_value.append(-1)
            print(f"Unable to save downloaded replay. Error: {e}")
            continue
        new_value.append(1)
        print(f"Downloaded replay {counter}")
        counter += 1
    return new_value