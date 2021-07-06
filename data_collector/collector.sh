#!/bin/bash
 
declare -a match_id_list=("6007589744" "6007490054" "6007393606" "6003945534" "6003877669" "6003824102" "6003725806" "6003682379" "5999791689" "5999726166" "5999656438" "5999590340" "5997018272" "5996901434" "5996813071" "5996655612" "5996563758" "5993267726" "5993209556" "5993131275" "5993077709" "5989186090" "5989118719" "5989044969" "5988951367" "5988891701" "5986132774" "5986047937" "5985926399" "5985867070" "5982664511" "5982601163" "5982525102" "5982478731" "5982439586" "5978375686" "5978296577" "5978231499" "5978224515" "5978081453" "5978027805" "5975053924" "5974971767" "5974944756" "5974803487" "5974717775" "5974631253" "5971437142" "5971372153" "5971296997" "5971241473" "5971185092" "5967166598" "5967102246" "5966984671" "5966933589" "5964092808" "5963977839" "5960320961" "5960254369" "5955652344" "5955536673" "5952214212" "5952123036" "5948043474" "5947984250" "5947921877" "5943290507" "5943215163" "6003863417" "5991708740" "5982534279" "5968616834" "5965367985" "5897702660" "5897463627" "5891735843" "5891624838" "5876785622" "5876695429" "5872814822" "5872679752" "5866577606" "5856603116" "5811658334" "5707370161" "5681817283" "5678303131" "5639623825" "5536512014" "5519238168" "5494563706" "5487119249" "5422285071" "5405526944" "5390160914" "5380057572" "5326872901" "5151162765" "6023742995" "6022177534" "6018212531" "6058247286" "6061542479" "6060268579" "6059898337" "6058324991")


for val in ${match_id_list[@]}; do
    echo $val
    # bzip2 -dk /home/hap/projects/dota2/player_identifier_Dota2/replay_downloader/downloads/$val.dem.bz2 
    # java -jar target/cursor_all.one-jar.jar /home/hap/projects/dota2/player_identifier_Dota2/replay_downloader/downloads/$val.dem 1
    java -jar target/unit_orders.one-jar.jar /home/hap/projects/dota2/player_identifier_Dota2/replay_downloader/downloads/$val.dem
    # java -jar target/info.one-jar.jar /home/hap/projects/dota2/player_identifier_Dota2/replay_downloader/downloads/$val.dem
done