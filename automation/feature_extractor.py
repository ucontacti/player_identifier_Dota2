import subprocess
import shutil

def replay_decompress(match_id_list):
    new_val = []
    for match in match_id_list:
        match = str(match)
        filename = "/media/hap/normal/replays/" + match + ".dem.bz2"
        try:    
            output = subprocess.run(['bzip2', '-dk', filename],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=True)
            shutil.move("/media/hap/normal/replays/" + match + ".dem", "/home/hap/projects/dota2/player_identifier_Dota2/replay_downloader/downloads/" + match + ".dem")
            print(output)
        except:
            new_value.append(-3)
            continue
        new_val.append(2)
    return new_val
def game_info(match_id_list):
    ## make sure the feature folder exists
    new_val = []
    for match in match_id_list:
        match = str(match)
        filename = "/home/hap/projects/dota2/player_identifier_Dota2/replay_downloader/downloads/" + match + ".dem"
        try:
            output = subprocess.run(['java', '-jar', '/home/hap/projects/dota2/player_identifier_Dota2/data_collector/target/info.one-jar.jar', filename],         
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True)
            print(output)
        except:
            new_value.append(-4)
            continue
        new_val.append(3)

    return new_val

def unit_order(match_id_list):
    ## make sure the feature folder exists
    new_val = []
    for match in match_id_list:
        match = str(match)
        filename = "/home/hap/projects/dota2/player_identifier_Dota2/replay_downloader/downloads/" + match + ".dem"
        try:
            output = subprocess.run(['java', '-jar', '/home/hap/projects/dota2/player_identifier_Dota2/data_collector/target/unit_orders.one-jar.jar', filename],         
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True)
            print(output)
        except:
            new_value.append(-5)
            continue
        new_val.append(4)
    return new_val

def cursor_data(match_id_list, tickrate = 1):
    ## make sure the feature folder exists
    new_val = []
    for match in match_id_list:
        match = str(match)
        filename = "/home/hap/projects/dota2/player_identifier_Dota2/replay_downloader/downloads/" + match + ".dem"
        try:
            output = subprocess.run(['java', '-jar', '/home/hap/projects/dota2/player_identifier_Dota2/data_collector/target/cursor_all.one-jar.jar', filename, str(tickrate)],         
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True)
            shutil.move("/home/hap/projects/dota2/player_identifier_Dota2/automation/features/" + match + "_cursor_tmp.csv", "/home/hap/projects/dota2/player_identifier_Dota2/automation/features/" + match + "_cursor_tmp_"  + tickrate + "_tick.csv")
            print(output)
        except:
            new_value.append(-6)
            continue
        
        new_val.append(5)
    return new_val, tickrate
