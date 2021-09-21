import requests, json
import urllib
# TODO: Add not download handler 
#       Add submit request


def submit_parse_request(match):
    pass

def replay_download(match_id_list):
    counter = 1
    new_value = []
    for match in match_id_list:
        match = str(match)
        stringurl = "https://api.opendota.com/api/replays?match_id=" + match
        raw_data = requests.get(stringurl, allow_redirects=True)
        json_data = json.loads(raw_data.content.decode())
        
        
        demo_url = "http://replay" + str(json_data[0]["cluster"]) + ".valve.net/570/" + str(json_data[0]["match_id"]) + "_" + str(json_data[0]["replay_salt"]) + ".dem.bz2"
        save_to = "/media/hap/normal/replays/" + match + ".dem.bz2"
        print("Getting ready to download replay " + str(counter) + " match_id: " + match)        
        try:
            urllib.request.urlretrieve(demo_url, save_to)
        except:
            new_value.append(-1)
            continue
        new_value.append(1)
        print("Downloaded replay " + str(counter))
        counter += 1
    return new_value