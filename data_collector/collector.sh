#!/bin/bash
 
declare -a match_id_list=("6157726715")


for val in ${match_id_list[@]}; do
    echo $val
    # bzip2 -dk /media/hap/normal/replays/$val.dem.bz2 
    java -jar target/cursor_all.one-jar.jar /home/hap/projects/dota2/player_identifier_Dota2/replay_downloader/downloads/$val.dem 10
    java -jar target/unit_orders.one-jar.jar /home/hap/projects/dota2/player_identifier_Dota2/replay_downloader/downloads/$val.dem
    java -jar target/info.one-jar.jar /home/hap/projects/dota2/player_identifier_Dota2/replay_downloader/downloads/$val.dem
done