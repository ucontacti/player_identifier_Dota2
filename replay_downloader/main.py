match_id_list = [
    "6033031231", 
    "6032259296",
    "6001694307",
    "6001298688",
    "6000983626",
    "6000062110",
    "5993525225",
    "5984859915",
    "5984783994",
    "5983263514",
    "5983057378",
    "5978072667",
    "5969008571",
    "5965724249",
    "5964370826",
    "5962442499",
    "5962365078",
    "5960698901",
    "5960538017",
    "5957598154",
    "5954355499",
    "5952464231",
    "5948576960",
    "5948149219",
    "5941846266",
    "5920558248",
    "5918806164",
    "5910767865",
    "5900276154",
    "5884457866",
    "5879077026",
    "5870990063",
    "5869865116",
    "5863880168",
    "5820576922",
    "5808584619",
    "5807080843",
    "5802540164",
    "5918913874",
    "5918806164",
    "5910767865",
    "5900276154",
    "5884457866",
    "5879077026",
    "5870990063",
    "5869865116",
    "5863880168",
    "5820576922",
    "5808584619",
]
import requests, json
import urllib
import socks
from sockshandler import SocksiPyHandler
from socks5_auth import socks5user, socks5pass

counter = 1
for match in match_id_list:
    stringurl = "https://api.opendota.com/api/replays?match_id=" + match
    raw_data = requests.get(stringurl, allow_redirects=True)
    json_data = json.loads(raw_data.content.decode())

    demo_url = "http://replay" + str(json_data[0]["cluster"]) + ".valve.net/570/" + str(json_data[0]["match_id"]) + "_" + str(json_data[0]["replay_salt"]) + ".dem.bz2"
    save_to = "downloads/" + match + ".dem.bz2"
    print("Getting ready to download replay " + str(counter) + " match_id: " + match)


    proxy = SocksiPyHandler(socks.SOCKS5, "amsterdam.nl.socks.nordhold.net", 1080, username=socks5user, password=socks5pass)
    opener = urllib.request.build_opener(proxy)
    urllib.request.install_opener(opener)
    
    requestt = urllib.request.urlopen("http://ip.42.pl/raw").read()
    print(requestt)
    print(demo_url)
    
    urllib.request.urlretrieve(demo_url, save_to)
    print("Downloaded replay " + str(counter))
    counter += 1