import subprocess
import shutil
import os

def replay_decompress(match_id_list):
    new_val = []
    for match in match_id_list:
        match = str(match)
        filename = "downloaded_replays/" + match + ".dem.bz2"
        try:    
            output = subprocess.run(['bzip2', '-dk', filename],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=True)
            shutil.move("downloaded_replays/" + match + ".dem", "downloaded_replays/" + match + ".dem")
            print(output)
        except:
            new_val.append(-3)
            continue
        new_val.append(2)
    return new_val
def game_info(match_id_list):
    ## make sure the feature folder exists
    new_val = []
    for match in match_id_list:
        match = str(match)
        filename = os.path.abspath("downloaded_replays/" + match + ".dem")
        try:
            output = subprocess.run(['java', '-jar', 'java/info.one-jar.jar', filename],         
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True)
            print(output)
        except:
            new_val.append(-4)
            continue
        new_val.append(3)

    return new_val

def unit_order(match_id_list):
    ## make sure the feature folder exists
    new_val = []
    for match in match_id_list:
        match = str(match)
        filename = "downloaded_replays/" + match + ".dem"
        try:
            output = subprocess.run(['java', '-jar', 'java/unit_orders.one-jar.jar', filename],         
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True)
            print(output)
        except:
            new_val.append(-5)
            continue
        new_val.append(4)
    return new_val

def item_change(match_id_list):
    ## make sure the feature folder exists
    new_val = []
    for match in match_id_list:
        match = str(match)
        filename = "downloaded_replays/" + match + ".dem"
        try:
            output = subprocess.run(['java', '-jar', 'java/item_change.one-jar.jar', filename],         
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True)
            print(output)
        except:
            new_val.append(-6)
            continue
        new_val.append(5)
    return new_val

def item_all(match_id_list):
    ## make sure the feature folder exists
    new_val = []
    for match in match_id_list:
        match = str(match)
        filename = os.path.abspath("downloads/" + match + ".dem")
        try:
            output = subprocess.run(['java', '-jar', 'java/item_all.one-jar.jar', filename], 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True)
            print(output)
        except:
            new_val.append(-7)
            continue
        new_val.append(6)
    return new_val

def cursor_data(match_id_list, tickrate = 1):
    ## make sure the feature folder exists
    new_val = []
    for match in match_id_list:
        match = str(match)
        filename = "downloaded_replays/" + match + ".dem"
        try:
            output = subprocess.run(['java', '-jar', 'java/cursor_all.one-jar.jar', filename, str(tickrate)],         
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True)
            shutil.move("java/features/" + match + "_cursor_tmp.csv", "java/features/" + match + "_cursor_tmp_"  + tickrate + "_tick.csv")
            print(output)
        except:
            new_val.append(-8)
            continue
        
        new_val.append(7)
    return new_val, tickrate
