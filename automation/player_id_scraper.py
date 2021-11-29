from bs4 import BeautifulSoup as soup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import time


DOTABUFF_HEROES=[
    # "https://www.dotabuff.com/heroes/abaddon/players",
    # "https://www.dotabuff.com/heroes/alchemist/players",
    # "https://www.dotabuff.com/heroes/ancient-apparition/players",
    # "https://www.dotabuff.com/heroes/anti-mage/players",
    # "https://www.dotabuff.com/heroes/arc-warden/players",
    # "https://www.dotabuff.com/heroes/axe/players",
    # "https://www.dotabuff.com/heroes/bane/players",
    # "https://www.dotabuff.com/heroes/batrider/players",
    # "https://www.dotabuff.com/heroes/beastmaster/players",
    # "https://www.dotabuff.com/heroes/bloodseeker/players",
    # "https://www.dotabuff.com/heroes/bounty-hunter/players",
    # "https://www.dotabuff.com/heroes/brewmaster/players",
    # "https://www.dotabuff.com/heroes/bristleback/players",
    # "https://www.dotabuff.com/heroes/broodmother/players",
    # "https://www.dotabuff.com/heroes/centaur-warrunner/players",
    # "https://www.dotabuff.com/heroes/chaos-knight/players",
    # "https://www.dotabuff.com/heroes/chen/players",
    # "https://www.dotabuff.com/heroes/clinkz/players",
    # "https://www.dotabuff.com/heroes/clockwerk/players",
    # "https://www.dotabuff.com/heroes/crystal-maiden/players",
    # "https://www.dotabuff.com/heroes/dark-seer/players",
    # "https://www.dotabuff.com/heroes/dark-willow/players",
    # "https://www.dotabuff.com/heroes/dawnbreaker/players",
    # "https://www.dotabuff.com/heroes/dazzle/players",
    # "https://www.dotabuff.com/heroes/death-prophet/players",
    # "https://www.dotabuff.com/heroes/disruptor/players",
    # "https://www.dotabuff.com/heroes/doom/players",
    # "https://www.dotabuff.com/heroes/dragon-knight/players",
    # "https://www.dotabuff.com/heroes/drow-ranger/players",
    # "https://www.dotabuff.com/heroes/earth-spirit/players",
    # "https://www.dotabuff.com/heroes/earthshaker/players",
    # "https://www.dotabuff.com/heroes/elder-titan/players",
    # "https://www.dotabuff.com/heroes/ember-spirit/players",
    # "https://www.dotabuff.com/heroes/enchantress/players",
    # "https://www.dotabuff.com/heroes/enigma/players",
    # "https://www.dotabuff.com/heroes/faceless-void/players",
    # "https://www.dotabuff.com/heroes/grimstroke/players",
    # "https://www.dotabuff.com/heroes/gyrocopter/players",
    # "https://www.dotabuff.com/heroes/hoodwink/players",
    # "https://www.dotabuff.com/heroes/huskar/players",
    # "https://www.dotabuff.com/heroes/invoker/players",
    # "https://www.dotabuff.com/heroes/io/players",
    # "https://www.dotabuff.com/heroes/jakiro/players",
    # "https://www.dotabuff.com/heroes/juggernaut/players",
    # "https://www.dotabuff.com/heroes/keeper-of-the-light/players",
    # "https://www.dotabuff.com/heroes/kunkka/players",
    # "https://www.dotabuff.com/heroes/legion-commander/players",
    # "https://www.dotabuff.com/heroes/leshrac/players",
    # "https://www.dotabuff.com/heroes/lich/players",
    # "https://www.dotabuff.com/heroes/lifestealer/players",
    # "https://www.dotabuff.com/heroes/lina/players",
    # "https://www.dotabuff.com/heroes/lion/players",
    # "https://www.dotabuff.com/heroes/lone-druid/players",
    # "https://www.dotabuff.com/heroes/luna/players",
    # "https://www.dotabuff.com/heroes/lycan/players",
    # "https://www.dotabuff.com/heroes/magnus/players",
    # "https://www.dotabuff.com/heroes/mars/players",
    # "https://www.dotabuff.com/heroes/medusa/players",
    # "https://www.dotabuff.com/heroes/meepo/players",
    # "https://www.dotabuff.com/heroes/mirana/players",
    # "https://www.dotabuff.com/heroes/monkey-king/players",
    # "https://www.dotabuff.com/heroes/morphling/players",
    # "https://www.dotabuff.com/heroes/naga-siren/players",
    # "https://www.dotabuff.com/heroes/natures-prophet/players",
    # "https://www.dotabuff.com/heroes/necrophos/players",
    # "https://www.dotabuff.com/heroes/night-stalker/players",
    # "https://www.dotabuff.com/heroes/nyx-assassin/players",
    # "https://www.dotabuff.com/heroes/ogre-magi/players",
    # "https://www.dotabuff.com/heroes/omniknight/players",
    # "https://www.dotabuff.com/heroes/oracle/players",
    # "https://www.dotabuff.com/heroes/outworld-destroyer/players",
    # "https://www.dotabuff.com/heroes/pangolier/players",
    # "https://www.dotabuff.com/heroes/phantom-assassin/players",
    # "https://www.dotabuff.com/heroes/phantom-lancer/players",
    # "https://www.dotabuff.com/heroes/phoenix/players",
    # "https://www.dotabuff.com/heroes/puck/players",
    # "https://www.dotabuff.com/heroes/pudge/players",
    # "https://www.dotabuff.com/heroes/pugna/players",
    # "https://www.dotabuff.com/heroes/queen-of-pain/players",
    # "https://www.dotabuff.com/heroes/razor/players",
    # "https://www.dotabuff.com/heroes/riki/players",
    # "https://www.dotabuff.com/heroes/rubick/players",
    # "https://www.dotabuff.com/heroes/sand-king/players",
    # "https://www.dotabuff.com/heroes/shadow-demon/players",
    # "https://www.dotabuff.com/heroes/shadow-fiend/players",
    # "https://www.dotabuff.com/heroes/shadow-shaman/players",
    # "https://www.dotabuff.com/heroes/silencer/players",
    # "https://www.dotabuff.com/heroes/skywrath-mage/players",
    # "https://www.dotabuff.com/heroes/slardar/players",
    # "https://www.dotabuff.com/heroes/slark/players",
    # "https://www.dotabuff.com/heroes/snapfire/players",
    # "https://www.dotabuff.com/heroes/sniper/players",
    # "https://www.dotabuff.com/heroes/spectre/players",
    # "https://www.dotabuff.com/heroes/spirit-breaker/players",
    # "https://www.dotabuff.com/heroes/storm-spirit/players",
    # "https://www.dotabuff.com/heroes/sven/players",
    # "https://www.dotabuff.com/heroes/techies/players",
    # "https://www.dotabuff.com/heroes/templar-assassin/players",
    # "https://www.dotabuff.com/heroes/terrorblade/players",
    # "https://www.dotabuff.com/heroes/tidehunter/players",
    # "https://www.dotabuff.com/heroes/timbersaw/players",
    "https://www.dotabuff.com/heroes/tinker/players",
    "https://www.dotabuff.com/heroes/tiny/players",
    "https://www.dotabuff.com/heroes/treant-protector/players",
    "https://www.dotabuff.com/heroes/troll-warlord/players",
    "https://www.dotabuff.com/heroes/tusk/players",
    "https://www.dotabuff.com/heroes/underlord/players",
    "https://www.dotabuff.com/heroes/undying/players",
    "https://www.dotabuff.com/heroes/ursa/players",
    "https://www.dotabuff.com/heroes/vengeful-spirit/players",
    "https://www.dotabuff.com/heroes/venomancer/players",
    "https://www.dotabuff.com/heroes/viper/players",
    "https://www.dotabuff.com/heroes/visage/players",
    "https://www.dotabuff.com/heroes/void-spirit/players",
    "https://www.dotabuff.com/heroes/warlock/players",
    "https://www.dotabuff.com/heroes/weaver/players",
    "https://www.dotabuff.com/heroes/windranger/players",
    "https://www.dotabuff.com/heroes/winter-wyvern/players",
    "https://www.dotabuff.com/heroes/witch-doctor/players",
    "https://www.dotabuff.com/heroes/wraith-king/players",
    "https://www.dotabuff.com/heroes/zeus/players",
]


def player_link(my_url):
    options = Options()
    options.page_load_strategy = 'normal'
    options.add_argument("--headless")
    driver = webdriver.Chrome("/home/hap/Downloads/chromedriver", options=options)
    driver.get(my_url)
    height = driver.execute_script("return document.body.scrollHeight/10")
    for i in range(10):
        driver.execute_script("window.scrollTo(0," + str(i*height) + ")")
        time.sleep(0.2)
    page_html = driver.page_source
    driver.close()
    page_soup = soup(page_html, "lxml")
    containers = page_soup.find_all("td", {"class":"cell-large"})
    counter = 0
    link_list = []
    for container in containers:
        link = container.find('a', href=True)['href']
        link = link[9:]
        link_list.append(link)
    return link_list
if __name__ == "__main__":
    new_players = []
    with open('player_index.txt', 'r') as filehandle:
        for line in filehandle:
            currentPlace = line[:-1]
            new_players.append(currentPlace)
    for hero_link in DOTABUFF_HEROES:
        print(hero_link)
        new_players.extend(player_link(hero_link))
        print("Done!")
    new_players = list(set(new_players))
    with open('player_index.txt', 'a') as filehandle:
        for listitem in new_players:
            filehandle.write('%s\n' % listitem)