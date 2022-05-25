package dota2identifier.item_all;


import com.google.protobuf.GeneratedMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import skadistats.clarity.model.Entity;
import skadistats.clarity.model.FieldPath;
import skadistats.clarity.model.state.EntityState;
import skadistats.clarity.model.StringTable;

import skadistats.clarity.processor.reader.OnMessage;
import skadistats.clarity.processor.reader.OnTickEnd;
import skadistats.clarity.processor.reader.OnTickStart;
import skadistats.clarity.processor.entities.Entities;
import skadistats.clarity.processor.entities.UsesEntities;
import skadistats.clarity.processor.runner.Context;
import skadistats.clarity.processor.runner.ControllableRunner;
import skadistats.clarity.processor.runner.SimpleRunner;
import skadistats.clarity.source.MappedFileSource;

import skadistats.clarity.processor.entities.OnEntityCreated;
import skadistats.clarity.processor.entities.OnEntityDeleted;
import skadistats.clarity.processor.entities.OnEntityPropertyCountChanged;
import skadistats.clarity.processor.entities.OnEntityUpdated;
import skadistats.clarity.processor.entities.OnEntityUpdatesCompleted;
import skadistats.clarity.processor.entities.UsesEntities;
import skadistats.clarity.processor.entities.OnEntityPropertyChanged;
import skadistats.clarity.event.Insert;
import skadistats.clarity.io.Util;

import skadistats.clarity.processor.stringtables.StringTables;
import skadistats.clarity.processor.stringtables.UsesStringTable;

import java.util.*;
import java.io.File;
import java.io.PrintWriter;

/**
 * Class to collect all the item data for each
 * player given a replay
 */

public class Main {

    private final Logger log = LoggerFactory.getLogger(Main.class.getPackage().getClass());

    public static final int RADIANT = 2;
    public static final int DIRE = 3;
    private boolean running = false;

    private boolean isHero(Entity e) {
        if (e.getDtClass().getDtName().equals("CDOTA_Unit_Hero_Beastmaster_Hawk"))
            return false;
        if (e.getDtClass().getDtName().equals("CDOTA_Unit_Hero_Beastmaster_Boar"))
            return false;
        return e.getDtClass().getDtName().startsWith("CDOTA_Unit_Hero");
    }
    public static boolean isPlayer(Entity e) {
        return e.getDtClass().getDtName().startsWith("CDOTAPlayer");
    }

    public boolean isGamePlayer(Entity e) {
    	if (isPlayer(e)) {
    		int playerTeamNum = getEntityProperty(e, "m_iTeamNum");
    		return playerTeamNum == RADIANT || playerTeamNum == DIRE;
    	}
    	return false;
    }

    public static Iterator<Entity> getEntities(Context ctx, String entityName) {
        if (ctx != null) {
            return ctx.getProcessor(Entities.class).getAllByDtName(entityName);
        }
        return null;
    }

    private int tick;
    private int item_updater_tick;
    private int cursor_updater_tick;
    private String update_hero_name;
    private int update_hero_id;
    private int count;
    private ArrayList<Entity> ent_list=new ArrayList<Entity>();
    private Hashtable<Integer, String> heroHashtbl = new Hashtable<Integer, String>();;
    private PrintWriter item_writer;
    private PrintWriter item_writer_updater;
    private int tick_rate;

    @Insert
    private Context ctx;

    public String getEntityNameByHandle(int id, Entities entities, StringTable stringTable) {
        Entity entity = entities.getByHandle(id);
        int index;
        if (entity == null) {
            return null;
        }
        else {
            index = entity.getProperty("m_pEntity.m_nameStringableIndex");
            return stringTable.getNameByIndex(index);
        }
    }

    @OnEntityCreated
    public void onCreated(Entity e) {
        if(e.getDtClass().getDtName().equals("CDOTAPlayer"))
        {
                ent_list.add(e);
        }
        if (isHero(e)) {
            Integer id = e.getProperty("m_iPlayerID");
            heroHashtbl.put(id, e.getDtClass().getDtName());
            ent_list.add(e);
        }     
    }

    public static String getTeamName(int team) {
        switch(team) {
            case 2: return "Radiant";
            case 3: return "Dire";
            default: return "";
        }
    }

    public static Entity getEntity(Context ctx, String entityName) {
        if (ctx != null) {
            return ctx.getProcessor(Entities.class).getByDtName(entityName);
        }
        return null;
    }

    public static <T> T resolveValue(Context ctx, String entityName, String pattern, int index, int team, int pos) {
        String fieldPathString = pattern
                .replaceAll("%i", Util.arrayIdxToString(index))
                .replaceAll("%t", Util.arrayIdxToString(team))
                .replaceAll("%p", Util.arrayIdxToString(pos));
        String compiledName = entityName.replaceAll("%n", getTeamName(team));
        Entity entity = getEntity(ctx, compiledName);
        FieldPath fieldPath = entity.getDtClass().getFieldPathForName(fieldPathString);

        return entity.getPropertyForFieldPath(fieldPath);
    }

    public static <T> T getEntityProperty(Entity e, String property) {
    	try {
            FieldPath f = e.getDtClass().getFieldPathForName(property);
            return e.getPropertyForFieldPath(f);
        } catch (Exception x) {
            return null;
        }
    }

    /**
     * The most important method that every tick for
     * every hero records the item slot from 1 to 9 of the
     * cursor on the screen
     * @param ctx
     * @param synthetic
     */

    @UsesEntities
    @UsesStringTable("EntityNames")
    @OnTickStart
    public void OnTickStart(Context ctx, boolean synthetic) {
        tick = ctx.getTick();
        if ((tick % tick_rate) == 0)
        {    
            Entities entities = ctx.getProcessor(Entities.class);
            StringTable stringTable = ctx.getProcessor(StringTables.class).forName("EntityNames");
            int itemId;
            for (Entity et: ent_list)
            {
                if(isHero(et))
                {
                    if(heroHashtbl.containsKey(et.getProperty("m_iPlayerID")))
                    {
                        if (((int) et.getProperty("m_hReplicatingOtherHeroModel") == 16777215)) 
                        {
                            StringBuilder sb = new StringBuilder();
                            sb.append(tick);
                            sb.append(',');
                            sb.append(heroHashtbl.get(et.getProperty("m_iPlayerID")));
                            for (int i=0; i<9; i++)
                            {
                                String item_slot = String.format("m_hItems.%04d", i);
                                int item_id = et.getProperty(item_slot);
                                String item_name = getEntityNameByHandle(item_id, entities, stringTable);
                                sb.append(',');
                                sb.append(item_name);
                            }
                            sb.append('\n');
                            item_writer.write(sb.toString());    
                        }
                    }
                }
            }
        }
    }
    
    public void run(String[] args) throws Exception {
        long tStart = System.currentTimeMillis();
        new SimpleRunner(new MappedFileSource(args[0])).runWith(this);
        long tMatch = System.currentTimeMillis() - tStart;
        running = false;
        log.info("total time taken: {}s", (tMatch) / 1000.0);
    }

    public void runControlled(String[] args) throws Exception {

        String rep_pwd = args[0];

        if (args.length == 1)
        {
            log.info("No tick rate specified. Sampling on 30 tirckrate.");
            tick_rate = 30;
        }
        else
        {
            tick_rate = Integer.parseInt(args[1]);
            if (tick_rate == 1 || tick_rate == 5 || tick_rate == 10 || tick_rate == 15 || tick_rate == 30)
            {
                ;
            }
            else
            {
                log.info("Tickrate should be one of the {1, 5, 10, 15, 30}. Sampling on everytick.");
                tick_rate = 30;
            }
        }
        String output = "features/" + rep_pwd.substring(rep_pwd.lastIndexOf("ds/") + 3, rep_pwd.lastIndexOf("s/") + 12) + "_item_all.csv";
        item_writer = new PrintWriter(new File(output));
        StringBuilder sb = new StringBuilder();
        sb.append("Tick");
        sb.append(',');
        sb.append("Hero");
        sb.append(',');
        sb.append("Item1");
        sb.append(',');
        sb.append("Item2");
        sb.append(',');
        sb.append("Item3");
        sb.append(',');
        sb.append("Item4");
        sb.append(',');
        sb.append("Item5");
        sb.append(',');
        sb.append("Item6");
        sb.append(',');
        sb.append("Item7");
        sb.append(',');
        sb.append("Item8");
        sb.append(',');
        sb.append("Item9");
        sb.append(',');
        sb.append('\n');
        item_writer.write(sb.toString());

        running = true;

        long tStart = System.currentTimeMillis();
        ControllableRunner cRunner = new ControllableRunner(new MappedFileSource(args[0])).runWith(this);
        for (int i = 0 ; i < 30000; i++) {
            cRunner.tick();
        }

        if (ctx != null) {
            Iterator<Entity> playerEntities = getEntities(ctx, "CDOTAPlayer");
            while(playerEntities.hasNext()) {
                Entity playerEntity = playerEntities.next();
                if (isGamePlayer(playerEntity))
                {
                    ent_list.add(playerEntity);
                    // int playerID = getEntityProperty(playerEntity, "m_iPlayerID");
                    // System.out.println(playerID);
                    // int heroID = resolveValue(ctx, "CDOTA_PlayerResource", "m_vecPlayerTeamData.%i.m_nSelectedHeroID", playerID, 0, 0);
                    // System.out.println(heroID);
                    // int selectedHero = resolveValue(ctx, "CDOTA_PlayerResource", "m_vecPlayerTeamData.%i.m_hSelectedHero", playerID, 0, 0);
                    // System.out.println(selectedHero);
                    // long steamID =  resolveValue(ctx, "CDOTA_PlayerResource", "m_vecPlayerData.%i.m_iPlayerSteamID", playerID, 0, 0);
                    // System.out.println(steamID);
                    // int teamNum = getEntityProperty(playerEntity, "m_iTeamNum");
                    // System.out.println(teamNum);
                }
            }
        }    

        while(!cRunner.isAtEnd()) {
            cRunner.tick();
        }
        cRunner.halt();

        long tMatch = System.currentTimeMillis() - tStart;
        log.info("total time taken: {}s", (tMatch) / 1000.0);    
        item_writer.close();
    }

    public static void main(String[] args) throws Exception {
        new Main().runControlled(args);
    }
}
