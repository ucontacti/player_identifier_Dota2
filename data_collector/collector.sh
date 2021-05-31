#!/bin/bash
 
declare -a match_id_list=("5915422921" "5915537315" "5915536193" "5915716111" "5916945544")
 
for val in ${match_id_list[@]}; do
    # bzip2 -dk /home/hap/projects/dota2/player_identifier_Dota2/replay_downloader/downloads/$val.dem.bz2 
    java -jar target/cursor_all.one-jar.jar /home/hap/projects/dota2/player_identifier_Dota2/replay_downloader/downloads/$val.dem
done