package dota2identifier.item_change;


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

/**
 * Class to collect all the item slot change 
 * data for each player given a replay
 */
public class Main {

    private final Logger log = LoggerFactory.getLogger(Main.class.getPackage().getClass());

    private boolean isHero(Entity e) {
        if (e.getDtClass().getDtName().equals("CDOTA_Unit_Hero_Beastmaster_Hawk"))
            return false;
        if (e.getDtClass().getDtName().equals("CDOTA_Unit_Hero_Beastmaster_Boar"))
            return false;
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
    private PrintWriter item_updater;

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

    @UsesEntities
    @UsesStringTable("EntityNames")
    @OnEntityPropertyChanged(classPattern = "CDOTA_Unit_Hero_.*", propertyPattern = "m_hItems.*")
    public void onItemChange(Context ctx, Entity et, FieldPath fp)
    {
        if (item_updater_tick != ctx.getTick() || !et.getDtClass().getDtName().equals(update_hero_name))
        {
            if (((int) et.getProperty("m_hReplicatingOtherHeroModel") == 16777215)) 
            {
                update_hero_name = et.getDtClass().getDtName();
                item_updater_tick = ctx.getTick();
                Entities entities = ctx.getProcessor(Entities.class);
                StringTable stringTable = ctx.getProcessor(StringTables.class).forName("EntityNames");
                int itemId;

                StringBuilder sb = new StringBuilder();
                sb.append(item_updater_tick + 1);
                sb.append(',');
                sb.append(heroHashtbl.get(et.getProperty("m_iPlayerID")));
                for (int i=0; i<9; i++)
                {
                    String item_slot = String.format("m_hItems.%04d", i);
                    int item_id = et.getProperty(item_slot);
                    String item_name = getEntityNameByHandle(item_id, entities, stringTable);
                    sb.append(',');
                    sb.append(item_name);
                    // sb.append(item_id);
                }
                sb.append('\n');
                item_updater.write(sb.toString());
            }
        }
    }


    public void run(String[] args) throws Exception {
        String rep_pwd = args[0];

        String output = "features/" + rep_pwd.substring(rep_pwd.lastIndexOf("ds/") + 3, rep_pwd.lastIndexOf("s/") + 12) + "_item_change_tmp.csv";
        
        item_updater = new PrintWriter(new File(output));
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
        item_updater.write(sb.toString());


        long tStart = System.currentTimeMillis();
        new SimpleRunner(new MappedFileSource(args[0])).runWith(this);
        long tMatch = System.currentTimeMillis() - tStart;
        log.info("total time taken: {}s", (tMatch) / 1000.0);
        item_updater.close();
    }



    public static void main(String[] args) throws Exception {
        new Main().run(args);
    }
}
