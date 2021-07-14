match_id_list = [
        "6086915544",
        "6087066937",
        "6087108350",
        "6087568230",
        "6087607461",
        "6087673366",
        "6088008579",
        "6088085832",
        "6088161692",
        "6087145277",
        "6087268442",
        "6087317203",
        "6087351438",
        "6088178730",
        "6086757362",
        "6086826130",
        "6086909970",
        "6087132554",
        "6087258618",
        "6087809321",
        "6087853096",
        "6087892765",
        "6087993662",
        "6088050000",
        "6088116258",
        "6087225724",
        "6087262154",
        "6087296156",
        "6087354717",
        "6087397552",
        "6087429335",
        "6087449764",
        "6087473685",
        "6087520900",
        "6087629657",
        "6087655740",
        "6086699958",
        "6086825343",
        "6087036870",
        "6087094858",
        "6088151751",
        "6086449969",
        "6087518459",
        "6087548874",
        "6087579429",
        "6087992218",
        "6086517517",
        "6086679219",
        "6086743997",
        "6086911619",
        "6087111410",
        "6087376668",
        "6087424427",
        "6087456905",
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
    save_to = "/media/hap/normal/replays/" + match + ".dem.bz2"
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