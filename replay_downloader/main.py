match_id_list = [
        "6068597079",
        "6068689145",
        "6068875600",
        "6068906300",
        "6075385871",
        "6075422209",
        "6075505310",
        "6076530801",
        "6076626173",
        "6076845187",
        "6076910924",
        "6078428435",
        "6078453867",
        "6078539630",
        "6079506227",
        "6079674173",
        "6079705496",
        "6081125184",
        "6081151993",
        "6081297154",
        "6081326175",
        "6081384753",
        "6082595901",
        "6084671518",
        "6084769058",
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


    # proxy = SocksiPyHandler(socks.SOCKS5, "amsterdam.nl.socks.nordhold.net", 1080, username=socks5user, password=socks5pass)
    # opener = urllib.request.build_opener(proxy)
    # urllib.request.install_opener(opener)
    
    # requestt = urllib.request.urlopen("http://ip.42.pl/raw").read()
    # print(requestt)
    print(demo_url)
    
    urllib.request.urlretrieve(demo_url, save_to)
    print("Downloaded replay " + str(counter))
    counter += 1