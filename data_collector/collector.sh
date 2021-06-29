#!/bin/bash
 
# declare -a match_id_list=("6033031231" "6032259296" "6001694307" "6001298688" "6000983626" "6000062110" "5993525225" "5984859915" "5984783994" "5983263514" "5983057378" "5978072667" "5969008571" "5965724249" "5964370826" "5962442499" "5962365078" "5960698901" "5960538017" "5957598154" "5954355499" "5952464231" "5948576960" "5948149219" "5941846266" "5920558248" "5918913874" "5918806164" "5910767865" "5900276154" "5884457866" "5879077026" "5870990063" "5869865116" "5863880168" "5820576922" "5808584619")
declare -a match_id_list=("6058247286" "6040639859" "6061542479" "6037154028" "6060268579" "6033930417" "6059898337" "6026361127" "6058324991")

        
for val in ${match_id_list[@]}; do
    echo $val
    bzip2 -dk /home/hap/projects/dota2/player_identifier_Dota2/replay_downloader/downloads/$val.dem.bz2 
    java -jar target/cursor_all.one-jar.jar /home/hap/projects/dota2/player_identifier_Dota2/replay_downloader/downloads/$val.dem 1
    java -jar target/unit_orders.one-jar.jar /home/hap/projects/dota2/player_identifier_Dota2/replay_downloader/downloads/$val.dem
    java -jar target/info.one-jar.jar /home/hap/projects/dota2/player_identifier_Dota2/replay_downloader/downloads/$val.dem
done