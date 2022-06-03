import subprocess
import shutil

REPLAY_PATH = "../resources/downloaded_replays"
JAVA_TARGET_PATH = "../../data_collector/target"
def replay_decompress(match_id_list):
    new_val = []
    for match in match_id_list:
        filename = f"{REPLAY_PATH}{match}.dem.bz2"
        try:
            output = subprocess.run(['bzip2', '-dk', filename],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=True)
            # shutil.move(f"{REPLAY_PATH}{match}.dem", f"{REPLAY_PATH}{match}.dem")
            print(output)
        except Exception as e:
            new_val.append(-3)
            print(f"Unable to decompress the replay demo. Error: {e}")
            continue
        new_val.append(2)
    return new_val
def game_info(match_id_list):
    ## make sure the feature folder exists
    new_val = []
    for match in match_id_list:
        filename = f"{REPLAY_PATH}/{match}.dem"
        try:
            output = subprocess.run(['java', '-jar', f'{JAVA_TARGET_PATH}/info.one-jar.jar', filename],         
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True)
            print(output)
        except Exception as e:
            new_val.append(-4)
            print(f"Unable to run java game_info. Error: {e}")
            continue
        new_val.append(3)

    return new_val

def unit_order(match_id_list):
    ## make sure the feature folder exists
    new_val = []
    for match in match_id_list:
        filename = f"{REPLAY_PATH}/{match}.dem"
        try:
            output = subprocess.run(['java', '-jar', f'{JAVA_TARGET_PATH}/unit_orders.one-jar.jar', filename],         
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True)
            print(output)
        except Exception as e:
            new_val.append(-5)
            print(f"Unable to run java unit_order. Error: {e}")
            continue
        new_val.append(4)
    return new_val

def item_change(match_id_list):
    ## make sure the feature folder exists
    new_val = []
    for match in match_id_list:
        filename = f"{REPLAY_PATH}/{match}.dem"
        try:
            output = subprocess.run(['java', '-jar', f'{JAVA_TARGET_PATH}/item_change.one-jar.jar', filename],         
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True)
            print(output)
        except Exception as e:
            new_val.append(-6)
            print(f"Unable to run java item_change. Error: {e}")
            continue
        new_val.append(5)
    return new_val

def item_all(match_id_list):
    ## make sure the feature folder exists
    new_val = []
    for match in match_id_list:
        filename = f"{REPLAY_PATH}/{match}.dem"
        try:
            output = subprocess.run(['java', '-jar', f'{JAVA_TARGET_PATH}/item_all.one-jar.jar', filename], 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True)
            print(output)
        except Exception as e:
            new_val.append(-7)
            print(f"Unable to run java item_all. Error: {e}")
            continue
        new_val.append(6)
    return new_val

def cursor_data(match_id_list, tickrate = 1):
    ## make sure the feature folder exists
    new_val = []
    for match in match_id_list:
        filename = f"{REPLAY_PATH}/{match}.dem"
        try:
            output = subprocess.run(['java', '-jar', f'{JAVA_TARGET_PATH}/cursor_all.one-jar.jar', filename, str(tickrate)],         
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True)
            shutil.move(f"../features/{match}_cursor_tmp.csv", f"../features/{match}_cursor_tmp_{tickrate}_tick.csv")
            print(output)
        except Exception as e:
            new_val.append(-8)
            print(f"Unable to run java cursor_all. Error: {e}")
            continue
        
        new_val.append(7)
    return new_val, tickrate
