package dota2identifier.unit_orders;


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

import skadistats.clarity.wire.common.proto.DotaUserMessages;


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
    private PrintWriter action_writer;

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

    @UsesStringTable("EntityNames")
    @UsesEntities
    @OnMessage(DotaUserMessages.CDOTAUserMsg_SpectatorPlayerUnitOrders.class)
    public void onMessage(Context ctx, DotaUserMessages.CDOTAUserMsg_SpectatorPlayerUnitOrders message) {
        tick = ctx.getTick();
        Entity et = ctx.getProcessor(Entities.class).getByIndex(message.getEntindex());
        Entities entities = ctx.getProcessor(Entities.class);
        StringTable stringTable = ctx.getProcessor(StringTables.class).forName("EntityNames");
        // Divide message type into actions
        StringBuilder sb = new StringBuilder();
        switch (message.getOrderType()) {
        case 1:
            sb.append(tick);
            sb.append(',');
            sb.append(heroHashtbl.get(et.getProperty("m_iPlayerID")));
            sb.append(',');
            sb.append('M');
            sb.append('\n');
            action_writer.write(sb.toString());
            break;
        case 2:
        case 4:
            sb.append(tick);
            sb.append(',');
            sb.append(heroHashtbl.get(et.getProperty("m_iPlayerID")));
            sb.append(',');
            sb.append('A');
            sb.append('\n');
            action_writer.write(sb.toString());
        case 8:
            sb.append(tick);
            sb.append(',');
            sb.append(heroHashtbl.get(et.getProperty("m_iPlayerID")));
            sb.append(',');
            sb.append('S');
            sb.append('\n');
            action_writer.write(sb.toString());
        }    
    //     if (message.hasAbilityId())
    //     {
    // //         log.info("orderType: {}, hasTargetIndex: {}, hasTargetIndex: {}, hasAbilityId: {}, hasPostion: {}",
    // //         // ctx.getTick(),
    // //         // message.getEntindex(),
    // //         message.getOrderType(),
    // //         message.hasTargetIndex(),
    // //         message.hasTargetIndex(),
    // //         message.hasAbilityId(),
    // //         message.hasPosition()
    // // );
    //     int ability_id = message.getAbilityId();
    //         String ability_name = getEntityNameByHandle(ability_id, entities, stringTable);
    //         if (ability_name != null)
    //             log.info("ability name {}: {}", ability_id, ability_name);
    //     }
    }


    public void run(String[] args) throws Exception {
        action_writer = new PrintWriter(new File("test_unit.csv"));
        StringBuilder sb = new StringBuilder();
        sb.append("Tick");
        sb.append(',');
        sb.append("Hero");
        sb.append(',');
        sb.append("Action");
        sb.append('\n');
        action_writer.write(sb.toString());


        long tStart = System.currentTimeMillis();
        new SimpleRunner(new MappedFileSource(args[0])).runWith(this);
        long tMatch = System.currentTimeMillis() - tStart;
        log.info("total time taken: {}s", (tMatch) / 1000.0);
        action_writer.close();
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
