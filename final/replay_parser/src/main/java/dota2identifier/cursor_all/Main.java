package dota2identifier.cursor_all;


import com.google.protobuf.GeneratedMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import skadistats.clarity.model.Entity;
import skadistats.clarity.model.FieldPath;
import skadistats.clarity.model.state.EntityState;
import skadistats.clarity.model.StringTable;
import skadistats.clarity.model.FieldPath;

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
 * Class to collect all the cursor data for each
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
    private PrintWriter mouse_writer;
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
        if (running)
        {
            if (isHero(e)) {
                Integer id = e.getProperty("m_iPlayerID");
                heroHashtbl.put(id, e.getDtClass().getDtName());
            }     
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
     * every hero records the tick, X and Y of the
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
                int player_id;
                if (et.hasProperty("m_nPlayerID"))
                {
                    player_id = et.getProperty("m_nPlayerID");
                }
                else
                {
                    player_id = et.getProperty("m_iPlayerID");
                }            
                if(isPlayer(et))
                {
                    if(heroHashtbl.containsKey(player_id))
                    {
                        StringBuilder sb = new StringBuilder();
                        sb.append(tick);
                        sb.append(',');
                        sb.append(heroHashtbl.get(player_id));
                        sb.append(',');
                        sb.append(et.getProperty("m_iCursor.0000"));
                        sb.append(',');
                        sb.append(et.getProperty("m_iCursor.0001"));
                        sb.append('\n');
                        mouse_writer.write(sb.toString());
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
            log.info("No tick rate specified. Sampling on everytick.");
            tick_rate = 1;
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
                tick_rate = 1;
            }
        }
        String output = "../features/" + rep_pwd.substring(rep_pwd.lastIndexOf("ys/") + 3, rep_pwd.lastIndexOf("s/") + 12) + "_cursor_tmp.csv";
        
        mouse_writer = new PrintWriter(new File(output));
        StringBuilder sb = new StringBuilder();
        sb.append("Tick");
        sb.append(',');
        sb.append("Hero");
        sb.append(',');
        sb.append("X");
        sb.append(',');
        sb.append("Y");
        sb.append('\n');
        mouse_writer.write(sb.toString());
        running = true;

        long tStart = System.currentTimeMillis();
        ControllableRunner cRunner = new ControllableRunner(new MappedFileSource(args[0])).runWith(this);
        for (int i = 0 ; i < 30000; i++) {
            cRunner.tick();
        }

        if (ctx != null) {
            Iterator<Entity> playerEntities = getEntities(ctx, "CDOTAPlayer");
            if (!playerEntities.hasNext())
            {
                playerEntities = getEntities(ctx, "CDOTAPlayerController");
            }
            while(playerEntities.hasNext()) {
                Entity playerEntity = playerEntities.next();
                if (isGamePlayer(playerEntity))
                {
                    ent_list.add(playerEntity);
                }
            }
        }    

        while(!cRunner.isAtEnd()) {
            cRunner.tick();
        }
        cRunner.halt();

        long tMatch = System.currentTimeMillis() - tStart;
        log.info("total time taken: {}s", (tMatch) / 1000.0);    
        mouse_writer.close();
    }


    public static void main(String[] args) throws Exception {
        new Main().runControlled(args);
    }

}
