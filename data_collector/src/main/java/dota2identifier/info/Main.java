package dota2identifier.info;

import skadistats.clarity.Clarity;
import skadistats.clarity.wire.common.proto.Demo.CDemoFileInfo;

import java.io.File;
import java.io.PrintWriter;

public class Main {
    
    
    public static String get_cdota_from_npc(String hero_name)
    {
        switch(hero_name)
        {
            case "npc_dota_hero_antimage": return "CDOTA_Unit_Hero_AntiMage";
            case "npc_dota_hero_axe": return "CDOTA_Unit_Hero_Axe";
            case "npc_dota_hero_bane": return "CDOTA_Unit_Hero_Bane";
            case "npc_dota_hero_bloodseeker": return "CDOTA_Unit_Hero_Bloodseeker";
            case "npc_dota_hero_crystal_maiden": return "CDOTA_Unit_Hero_CrystalMaiden";
            case "npc_dota_hero_drow_ranger": return "CDOTA_Unit_Hero_DrowRanger";
            case "npc_dota_hero_earthshaker": return "CDOTA_Unit_Hero_Earthshaker";
            case "npc_dota_hero_juggernaut": return "CDOTA_Unit_Hero_Juggernaut";
            case "npc_dota_hero_mirana": return "CDOTA_Unit_Hero_Mirana";
            case "npc_dota_hero_nevermore": return "CDOTA_Unit_Hero_Nevermore";
            case "npc_dota_hero_morphling": return "CDOTA_Unit_Hero_Morphling";
            case "npc_dota_hero_phantom_lancer": return "CDOTA_Unit_Hero_PhantomLancer";
            case "npc_dota_hero_puck": return "CDOTA_Unit_Hero_Puck";
            case "npc_dota_hero_pudge": return "CDOTA_Unit_Hero_Pudge";
            case "npc_dota_hero_razor": return "CDOTA_Unit_Hero_Razor";
            case "npc_dota_hero_sand_king": return "CDOTA_Unit_Hero_SandKing";
            case "npc_dota_hero_storm_spirit": return "CDOTA_Unit_Hero_StormSpirit";
            case "npc_dota_hero_sven": return "CDOTA_Unit_Hero_Sven";
            case "npc_dota_hero_tiny": return "CDOTA_Unit_Hero_Tiny";
            case "npc_dota_hero_vengefulspirit": return "CDOTA_Unit_Hero_VengefulSpirit";
            case "npc_dota_hero_windrunner": return "CDOTA_Unit_Hero_Windrunner";
            case "npc_dota_hero_zuus": return "CDOTA_Unit_Hero_Zuus";
            case "npc_dota_hero_kunkka": return "CDOTA_Unit_Hero_Kunkka";
            case "npc_dota_hero_lina": return "CDOTA_Unit_Hero_Lina";
            case "npc_dota_hero_lich": return "CDOTA_Unit_Hero_Lich";
            case "npc_dota_hero_lion": return "CDOTA_Unit_Hero_Lion";
            case "npc_dota_hero_shadow_shaman": return "CDOTA_Unit_Hero_ShadowShaman";
            case "npc_dota_hero_slardar": return "CDOTA_Unit_Hero_Slardar";
            case "npc_dota_hero_tidehunter": return "CDOTA_Unit_Hero_Tidehunter";
            case "npc_dota_hero_witch_doctor": return "CDOTA_Unit_Hero_WitchDoctor";
            case "npc_dota_hero_riki": return "CDOTA_Unit_Hero_Riki";
            case "npc_dota_hero_enigma": return "CDOTA_Unit_Hero_Enigma";
            case "npc_dota_hero_tinker": return "CDOTA_Unit_Hero_Tinker";
            case "npc_dota_hero_sniper": return "CDOTA_Unit_Hero_Sniper";
            case "npc_dota_hero_necrolyte": return "CDOTA_Unit_Hero_Necrolyte";
            case "npc_dota_hero_warlock": return "CDOTA_Unit_Hero_Warlock";
            case "npc_dota_hero_beastmaster": return "CDOTA_Unit_Hero_Beastmaster";
            case "npc_dota_hero_queenofpain": return "CDOTA_Unit_Hero_QueenOfPain";
            case "npc_dota_hero_venomancer": return "CDOTA_Unit_Hero_Venomancer";
            case "npc_dota_hero_faceless_void": return "CDOTA_Unit_Hero_FacelessVoid";
            case "npc_dota_hero_skeleton_king": return "CDOTA_Unit_Hero_SkeletonKing";
            case "npc_dota_hero_death_prophet": return "CDOTA_Unit_Hero_DeathProphet";
            case "npc_dota_hero_phantom_assassin": return "CDOTA_Unit_Hero_PhantomAssassin";
            case "npc_dota_hero_pugna": return "CDOTA_Unit_Hero_Pugna";
            case "npc_dota_hero_templar_assassin": return "CDOTA_Unit_Hero_TemplarAssassin";
            case "npc_dota_hero_viper": return "CDOTA_Unit_Hero_Viper";
            case "npc_dota_hero_luna": return "CDOTA_Unit_Hero_Luna";
            case "npc_dota_hero_dragon_knight": return "CDOTA_Unit_Hero_DragonKnight";
            case "npc_dota_hero_dazzle": return "CDOTA_Unit_Hero_Dazzle";
            case "npc_dota_hero_rattletrap": return "CDOTA_Unit_Hero_Rattletrap";
            case "npc_dota_hero_leshrac": return "CDOTA_Unit_Hero_Leshrac";
            case "npc_dota_hero_furion": return "CDOTA_Unit_Hero_Furion";
            case "npc_dota_hero_life_stealer": return "CDOTA_Unit_Hero_Life_Stealer";
            case "npc_dota_hero_dark_seer": return "CDOTA_Unit_Hero_DarkSeer";
            case "npc_dota_hero_clinkz": return "CDOTA_Unit_Hero_Clinkz";
            case "npc_dota_hero_omniknight": return "CDOTA_Unit_Hero_Omniknight";
            case "npc_dota_hero_enchantress": return "CDOTA_Unit_Hero_Enchantress";
            case "npc_dota_hero_huskar": return "CDOTA_Unit_Hero_Huskar";
            case "npc_dota_hero_night_stalker": return "CDOTA_Unit_Hero_NightStalker";
            case "npc_dota_hero_broodmother": return "CDOTA_Unit_Hero_Broodmother";
            case "npc_dota_hero_bounty_hunter": return "CDOTA_Unit_Hero_BountyHunter";
            case "npc_dota_hero_weaver": return "CDOTA_Unit_Hero_Weaver";
            case "npc_dota_hero_jakiro": return "CDOTA_Unit_Hero_Jakiro";
            case "npc_dota_hero_batrider": return "CDOTA_Unit_Hero_Batrider";
            case "npc_dota_hero_chen": return "CDOTA_Unit_Hero_Chen";
            case "npc_dota_hero_spectre": return "CDOTA_Unit_Hero_Spectre";
            case "npc_dota_hero_ancient_apparition": return "CDOTA_Unit_Hero_AncientApparition";
            case "npc_dota_hero_doom_bringer": return "CDOTA_Unit_Hero_DoomBringer";
            case "npc_dota_hero_ursa": return "CDOTA_Unit_Hero_Ursa";
            case "npc_dota_hero_spirit_breaker": return "CDOTA_Unit_Hero_SpiritBreaker";
            case "npc_dota_hero_gyrocopter": return "CDOTA_Unit_Hero_Gyrocopter";
            case "npc_dota_hero_alchemist": return "CDOTA_Unit_Hero_Alchemist";
            case "npc_dota_hero_invoker": return "CDOTA_Unit_Hero_Invoker";
            case "npc_dota_hero_silencer": return "CDOTA_Unit_Hero_Silencer";
            case "npc_dota_hero_obsidian_destroyer": return "CDOTA_Unit_Hero_Obsidian_Destroyer";
            case "npc_dota_hero_lycan": return "CDOTA_Unit_Hero_Lycan";
            case "npc_dota_hero_brewmaster": return "CDOTA_Unit_Hero_Brewmaster";
            case "npc_dota_hero_shadow_demon": return "CDOTA_Unit_Hero_Shadow_Demon";
            case "npc_dota_hero_lone_druid": return "CDOTA_Unit_Hero_LoneDruid";
            case "npc_dota_hero_chaos_knight": return "CDOTA_Unit_Hero_ChaosKnight";
            case "npc_dota_hero_meepo": return "CDOTA_Unit_Hero_Meepo";
            case "npc_dota_hero_treant": return "CDOTA_Unit_Hero_Treant";
            case "npc_dota_hero_ogre_magi": return "CDOTA_Unit_Hero_Ogre_Magi";
            case "npc_dota_hero_undying": return "CDOTA_Unit_Hero_Undying";
            case "npc_dota_hero_rubick": return "CDOTA_Unit_Hero_Rubick";
            case "npc_dota_hero_disruptor": return "CDOTA_Unit_Hero_Disruptor";
            case "npc_dota_hero_nyx_assassin": return "CDOTA_Unit_Hero_Nyx_Assassin";
            case "npc_dota_hero_naga_siren": return "CDOTA_Unit_Hero_Naga_Siren";
            case "npc_dota_hero_keeper_of_the_light": return "CDOTA_Unit_Hero_KeeperOfTheLight";
            case "npc_dota_hero_wisp": return "CDOTA_Unit_Hero_Wisp";
            case "npc_dota_hero_visage": return "CDOTA_Unit_Hero_Visage";
            case "npc_dota_hero_slark": return "CDOTA_Unit_Hero_Slark";
            case "npc_dota_hero_medusa": return "CDOTA_Unit_Hero_Medusa";
            case "npc_dota_hero_troll_warlord": return "CDOTA_Unit_Hero_TrollWarlord";
            case "npc_dota_hero_centaur": return "CDOTA_Unit_Hero_Centaur";
            case "npc_dota_hero_magnataur": return "CDOTA_Unit_Hero_Magnataur";
            case "npc_dota_hero_shredder": return "CDOTA_Unit_Hero_Shredder";
            case "npc_dota_hero_bristleback": return "CDOTA_Unit_Hero_Bristleback";
            case "npc_dota_hero_tusk": return "CDOTA_Unit_Hero_Tusk";
            case "npc_dota_hero_skywrath_mage": return "CDOTA_Unit_Hero_Skywrath_Mage";
            case "npc_dota_hero_abaddon": return "CDOTA_Unit_Hero_Abaddon";
            case "npc_dota_hero_elder_titan": return "CDOTA_Unit_Hero_Elder_Titan";
            case "npc_dota_hero_legion_commander": return "CDOTA_Unit_Hero_Legion_Commander";
            case "npc_dota_hero_ember_spirit": return "CDOTA_Unit_Hero_EmberSpirit";
            case "npc_dota_hero_earth_spirit": return "CDOTA_Unit_Hero_EarthSpirit";
            case "npc_dota_hero_abyssal_underlord": return "CDOTA_Unit_Hero_AbyssalUnderlord";
            case "npc_dota_hero_terrorblade": return "CDOTA_Unit_Hero_Terrorblade";
            case "npc_dota_hero_void_spirit": return "CDOTA_Unit_Hero_Void_Spirit";
            case "npc_dota_hero_phoenix": return "CDOTA_Unit_Hero_Phoenix";
            case "npc_dota_hero_techies": return "CDOTA_Unit_Hero_Techies";
            case "npc_dota_hero_oracle": return "CDOTA_Unit_Hero_Oracle";
            case "npc_dota_hero_monkey_king": return "CDOTA_Unit_Hero_MonkeyKing";
            case "npc_dota_hero_pangolier": return "CDOTA_Unit_Hero_Pangolier";
            case "npc_dota_hero_dark_willow": return "CDOTA_Unit_Hero_DarkWillow";
            case "npc_dota_hero_grimstroke": return "CDOTA_Unit_Hero_Grimstroke";
            case "npc_dota_hero_mars": return "CDOTA_Unit_Hero_Mars";
            case "npc_dota_hero_snapfire": return "CDOTA_Unit_Hero_Snapfire";
            case "npc_dota_hero_hoodwink": return "CDOTA_Unit_Hero_Hoodwink";
            case "npc_dota_hero_dawnbreaker": return "CDOTA_Unit_Hero_Dawnbreaker";
            case "npc_dota_hero_arc_warden": return "CDOTA_Unit_Hero_ArcWarden";
            case "npc_dota_hero_winter_wyvern": return "CDOTA_Unit_Hero_Winter_Wyvern";
            case "npc_dota_hero_marci": return "CDOTA_Unit_Hero_Marci";            
            case "npc_dota_hero_primal_beast": return "CDOTA_Unit_Hero_PrimalBeast";
            default: return "Unknown";
        }
    }


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
            sb.append(get_cdota_from_npc(info.getGameInfo().getDota().getPlayerInfoList().get(i).getHeroName()));
            sb.append('\n');
            info_writer.write(sb.toString());
        }

        info_writer.close();
    }
}
