#!/usr/bin/env python3

"""
Scripting functions of the game.
All functions should start with the env parameter, like being methods of the game class.
"""

import sys
sys.dont_write_bytecode=True

def script_core_set_start_positions(env,*args,**kwargs)->bool:
    env.set_start_positions(start=[0x00,9,6,0],checkpoint=[0x00,9,6,0])

def script_core_set_automatic_map_ids(env,*args,**kwargs)->bool:
    env.set_automatic_scripted_map_ids([])
    return False

def script_core_set_extra_event_names(env,*args,**kwargs)->bool:
    event_names=[]
    env.set_extra_event_names(event_names)
    return False

def script_core_automatic_with_npc(env,map_id:int=0,*args,**kwargs)->bool:
    return False

def script_core_automatic_without_npc(env,map_id:int=0,*args,**kwargs)->bool:
    return False

def script_core_automatic_map_transition(env,new_map_id:int,global_warped:bool,teleported:bool,*args,**kwargs)->bool:
    return False

def script_core_reset(env,*args,**kwargs)->bool:
    return False

def script_core_load_custom_save_from_starting_event(env,name:str="",*args,**kwargs)->bool:
    return False

def script_npc_text(env,*args,**kwargs)->bool:
    env.game_state["text_type"]=1
    return False

def script_invert_direction_text(env,*args,**kwargs)->bool:
    env.add_forced_movement_invert_direction()
    env.game_state["text_type"]=1
    return False

def script_heal(env,*args,**kwargs)->bool:
    env.party_heal()
    return False

def script_heal_checkpoint(env,*args,**kwargs)->bool:
    env.set_checkpoint_map_position(*env.game_state["player_coordinates_data"][:4])
    script_heal(env)

def script_exiting_first_town(env,*args,**kwargs)->bool:
    env.activate_event_reward_flag("exiting_first_town")
    return True

def script_powerup_debush(env,*args,**kwargs)->bool:
    if env.get_event_flag("powerup_debush")==0:
        env.activate_event_reward_flag("powerup_debush")
    return False

def script_club1_leader(env,*args,**kwargs)->bool:
    if env.get_event_flag("medal1")==0:
        env.activate_event_reward_flag("medal1")
    return False

def script_club2_leader(env,*args,**kwargs)->bool:
    if env.get_event_flag("medal2")==0:
        env.activate_event_reward_flag("medal2")
    return False

def script_powerup_swim(env,*args,**kwargs)->bool:
    if env.get_event_flag("powerup_swim")==0:
        env.activate_event_reward_flag("powerup_swim")
    return False

def script_club3_leader(env,*args,**kwargs)->bool:
    if env.get_event_flag("medal3")==0:
        env.activate_event_reward_flag("medal3")
    return False

def script_powerup_cross_whirlpool(env,*args,**kwargs)->bool:
    if env.get_event_flag("powerup_cross_whirlpool")==0:
        env.activate_event_reward_flag("powerup_cross_whirlpool")
    return False

def script_powerup_cross_waterfall(env,*args,**kwargs)->bool:
    if env.get_event_flag("powerup_waterfall")==0:
        env.activate_event_reward_flag("powerup_waterfall")
    return False

def script_club4_leader(env,*args,**kwargs)->bool:
    if env.get_event_flag("medal4")==0:
        env.activate_event_reward_flag("medal4")
    return False

def script_club5_leader(env,*args,**kwargs)->bool:
    if env.get_event_flag("medal5")==0:
        env.activate_event_reward_flag("medal5")
    return False

def script_powerup_teleport(env,*args,**kwargs)->bool:
    if env.get_event_flag("powerup_teleport")==0:
        env.activate_event_reward_flag("powerup_teleport")
    return False

def script_city2_engineer(env,*args,**kwargs)->bool:
    if env.get_event_flag("medal5")==0:
#        env.modify_npc_configs("path9_npc_guard_building_site",{"x":8})
        if env.get_event_flag("engineer_pass")==0:
            env.activate_event_reward_flag("engineer_pass")
    return False

def script_path9_guard(env,*args,**kwargs)->bool:
    if env.get_event_flag("engineer_pass")>0:
        env.modify_npc_configs("path9_npc_guard_building_site",{"x":8})
        if env.get_event_flag("engineer_guard")==0:
            env.activate_event_reward_flag("engineer_guard")
        return True
    return script_invert_direction_text(env)

def script_club6_leader(env,*args,**kwargs)->bool:
    if env.get_event_flag("medal6")==0:
        env.activate_event_reward_flag("medal6")
    return False

def script_powerup_break_rock(env,*args,**kwargs)->bool:
    if env.get_event_flag("powerup_break_rock")==0:
        env.activate_event_reward_flag("powerup_break_rock")
    return False

def script_powerup_break_frozen_rock(env,*args,**kwargs)->bool:
    if env.get_event_flag("powerup_break_frozen_rock")==0:
        env.activate_event_reward_flag("powerup_break_frozen_rock")
    return False

def script_club7_leader(env,*args,**kwargs)->bool:
    if env.get_event_flag("medal7")==0:
        env.activate_event_reward_flag("medal7")
    return False

def script_powerup_mountain_climb(env,*args,**kwargs)->bool:
    if env.get_event_flag("powerup_mountain_climb")==0:
        env.activate_event_reward_flag("powerup_mountain_climb")
    return False

def script_club8_leader(env,*args,**kwargs)->bool:
    if env.get_event_flag("medal8")==0:
        env.activate_event_reward_flag("medal8")
    return False

def script_medals_check(env,*args,**kwargs)->bool:
    if env.get_event_flag("medal8")>0:
        if env.get_event_flag("all_medals_check")==0:
            env.activate_event_reward_flag("all_medals_check")
        return True
    return script_invert_direction_text(env)

def script_powerups_check(env,*args,**kwargs)->bool:
    if env.get_event_flag("powerup_teleport")>0 and env.get_event_flag("powerup_mountain_climb")>0:
        if env.get_event_flag("all_powerups_check")==0:
            env.activate_event_reward_flag("all_powerups_check")
        return True
    return script_invert_direction_text(env)

def script_entering_final_area(env,*args,**kwargs)->bool:
    if env.get_event_flag("all_powerups_check")==0:
        if env.get_event_flag("entering_final_area")==0:
            env.activate_event_reward_flag("entering_final_area")
        return False
    return script_invert_direction_text(env)

def script_final_champion(env,*args,**kwargs)->bool:
    if env.get_event_flag("final_champion")==0:
        env.activate_event_reward_flag("final_champion")
    env.set_checkpoint_map_position(*env.first_checkpoint_place)
    env.force_teleport_to(*env.first_checkpoint_place)
    return False
