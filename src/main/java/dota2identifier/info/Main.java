package dota2identifier.info;

import skadistats.clarity.Clarity;
import skadistats.clarity.wire.common.proto.Demo.CDemoFileInfo;

import java.io.File;
import java.io.PrintWriter;


public class Main {
    
    public static void main(String[] args) throws Exception {
        String test = args[0];
        String output = "features/" + test.substring(test.lastIndexOf("ds/") + 3, test.lastIndexOf("s/") + 12) + "_info.csv";
        PrintWriter info_writer = new PrintWriter(new File(output));
        StringBuilder sb = new StringBuilder();
        sb.append("SteamId");
        sb.append(',');
        sb.append("Hero");
        sb.append('\n');
        info_writer.write(sb.toString());

        CDemoFileInfo info = Clarity.infoForFile(args[0]);
        for (int i = 0; i < 10; i++)
        {
            sb = new StringBuilder();
            sb.append(info.getGameInfo().getDota().getPlayerInfoList().get(i).getSteamid());
            sb.append(',');
            sb.append(info.getGameInfo().getDota().getPlayerInfoList().get(i).getHeroName());
            sb.append('\n');
            info_writer.write(sb.toString());
        }

        info_writer.close();
    }
}
