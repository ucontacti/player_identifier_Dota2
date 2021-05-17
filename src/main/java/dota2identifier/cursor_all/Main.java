package dota2identifier.cursor_all;


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

import skadistats.clarity.processor.stringtables.StringTables;
import skadistats.clarity.processor.stringtables.UsesStringTable;

import java.util.ArrayList;
import java.util.Hashtable;
import java.io.File;
import java.io.PrintWriter;

public class Main {

    private final Logger log = LoggerFactory.getLogger(Main.class.getPackage().getClass());

    // TODO: Improve hero recognition
    private boolean isHero(Entity e) {
        return e.getDtClass().getDtName().startsWith("CDOTA_Unit_Hero");
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
    private PrintWriter mouse_writer_updater;
    private PrintWriter item_writer;
    private PrintWriter item_writer_updater;

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
        // if(isHero(e))
        //     if (0 <= Integer.parseInt(e.getProperty("m_iPlayerID"))  && Integer.parseInt(e.getProperty("m_iPlayerID")) <= 9)
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


    @UsesEntities
    @UsesStringTable("EntityNames")
    @OnTickStart
    public void OnTickEnd(Context ctx, boolean synthetic) {
        tick = ctx.getTick();
        Entities entities = ctx.getProcessor(Entities.class);
        StringTable stringTable = ctx.getProcessor(StringTables.class).forName("EntityNames");
        int itemId;

        for (Entity et: ent_list)
        {
            // if(heroHashtbl.containsKey(et.getProperty("m_iPlayerID")))
            // {
            //     log.info("tick {}, entity {}: {}, {}", tick, heroHashtbl.get(et.getProperty("m_iPlayerID")), et.getProperty("m_iCursor.0000"), et.getProperty("m_iCursor.0001"));
            // }
            if(et.getDtClass().getDtName().equals("CDOTAPlayer"))
            {
                if(heroHashtbl.containsKey(et.getProperty("m_iPlayerID")))
                {
                    // log.info("tick {}, entity {}: {}, {}", tick, heroHashtbl.get(et.getProperty("m_iPlayerID")), et.getProperty("m_iCursor.0000"), et.getProperty("m_iCursor.0001"));
                    // log.info("tick {}, entity {}: {} {}", tick, et.getProperty("m_iPlayerID"), et.getProperty("m_iCursor.0000"), et.getProperty("m_iCursor.0001"));
                    StringBuilder sb = new StringBuilder();
                    sb.append(tick);
                    sb.append(',');
                    sb.append(heroHashtbl.get(et.getProperty("m_iPlayerID")));
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
    
    public void run(String[] args) throws Exception {
        mouse_writer = new PrintWriter(new File("test_cursor_all.csv"));
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

        long tStart = System.currentTimeMillis();
        new SimpleRunner(new MappedFileSource(args[0])).runWith(this);
        long tMatch = System.currentTimeMillis() - tStart;
        log.info("total time taken: {}s", (tMatch) / 1000.0);
        mouse_writer.close();
    }

    public void runControlled(String[] args) throws Exception {
        long tStart = System.currentTimeMillis();
        ControllableRunner runner = new ControllableRunner(new MappedFileSource(args[0])).runWith(this);
        while(!runner.isAtEnd()) {
            runner.tick();
        }
        long tMatch = System.currentTimeMillis() - tStart;
        log.info("total time taken: {}s", (tMatch) / 1000.0);
        
    }


    public static void main(String[] args) throws Exception {
        new Main().run(args);
    }

}
