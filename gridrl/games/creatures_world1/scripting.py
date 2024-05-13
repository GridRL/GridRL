#!/usr/bin/env python3

"""
Scripting functions of the game.
All functions should start with the env parameter, like being methods of the game class.
"""

import sys
import numpy as np
sys.dont_write_bytecode=True

#__all__=[]

def script_core_set_start_positions(env,*args,**kwargs)->bool:
    """@."""
    env.set_start_positions(start=[0x26,6,3,1],checkpoint=[0x25,4,6,1])

def script_core_set_automatic_map_ids(env,*args,**kwargs)->bool:
    """@."""
    trek_map_ids=[0xD9,0xDA,0xDB,0xDC,0xDD,0xDE,0xDF,0xE0,0xE1]
    elevators_map_ids=[0x7F,0xCB,0xEC]
    env.set_automatic_scripted_map_ids(trek_map_ids+elevators_map_ids)
    return False

def script_core_set_extra_event_names(env,*args,**kwargs)->bool:
    """@."""
    event_names=[]
    env.set_extra_event_names(event_names)
    return False

def script_core_automatic_with_npc(env,map_id:int=0,*args,**kwargs)->bool:
    """@."""
    ret=False
    if env.cached_is_trek_map(map_id):
        ret=script_trek_timer(env)
    elif map_id==0xEC:
        ret=script_corporate_building_elevator(env)
    return ret

def script_core_automatic_without_npc(env,map_id:int=0,*args,**kwargs)->bool:
    """@."""
    ret=False
    if env.cached_is_trek_map(map_id):
        ret=script_trek_timer(env)
    elif map_id==0x7F:
        ret=script_commercial_city_shop_elevator(env)
    elif not env.using_npcs or len(env.npcs_data)==0:
        if map_id==0xCB:
            ret=script_evils_refuge_elevator(env)
        elif map_id==0xEC:
            ret=script_corporate_building_elevator(env)
    return ret

def script_core_automatic_map_transition(env,new_map_id:int,global_warped:bool,teleported:bool)->bool:
    """@."""
    if not global_warped or teleported:
        prev_map_id=env.get_current_map_id()
        if (teleported or prev_map_id==0x9C) and env.cached_is_trek_map(new_map_id):
            script_trek_start(env,True)
        elif (teleported or prev_map_id==0x08) and new_map_id==0xA5:
            script_vulcanic_refuge_doors_switches_initialize(env)
    return False

def script_core_reset(env,*args,**kwargs)->bool:
    """@."""
    script_club_puzzle_initialize(env)
    return False

def script_core_load_custom_save_from_starting_event(env,name:str="",*args,**kwargs)->bool:
    """@."""
    return False

def script_npc_text_vanish_chance(env,*args,**kwargs)->bool:
    """@."""
    env.show_npc_text()
    interactions=env.get_npc_configs(env.last_npc_name)["interactions"]
    if interactions>1 and (np.random.rand()<0.5 or interactions>2):
        env.modify_npc_configs(env.last_npc_name,{"state":0})
    return True

def script_npc_text(env,*args,**kwargs)->bool:
    """@."""
    env.show_npc_text()
    return False

def script_npc_text_once(env,*args,**kwargs)->bool:
    """@."""
    env.show_npc_text()
    env.modify_npc_configs(env.last_npc_name,{"state":0})
    return True

def script_invert_direction_text(env,*args,**kwargs)->bool:
    """@."""
    env.add_forced_movement_invert_direction()
    env.show_npc_text()
    return False

def script_ledge_down(env,*args,**kwargs)->bool:
    """@."""
    if env.game_state["player_coordinates_data"][3]==0:
        env.add_forced_movement_down()
    else:
        env.step_game_invert_direction_keep_facing()
    return False

def script_npc_battle(env,*args,**kwargs)->bool:
    """@."""
    if len(args)==0 or env.get_event_flag("start_decision")==0:
        return False
    levels=np.asarray(args[0])[:,0]
    if len(levels)==0:
        return True
    moves_penalty=float(kwargs.get("moves_penalty",0.67))
    if max(levels)<16 and moves_penalty>0.5:
        moves_penalty*=1-0.1*(len(levels)-1)
    return env.headless_battle_npc(level=max(levels),num=len(levels),moves_penalty=moves_penalty)

def script_npc_battle_vanish(env,*args,**kwargs)->bool:
    """@."""
    if len(args)==0:
        return False
    npc_name=env.last_npc_name
    ret=script_npc_battle(env,*args)
    if ret:
        env.modify_npc_configs(npc_name,{"state":0})
    return ret

def script_gift_creature(env,*args,**kwargs)->bool:
    """@."""
    env.show_npc_text()
    interactions=env.get_npc_configs(env.last_npc_name)["interactions"]
    if interactions==1:
        env.set_first_party_creature(*args[0])
    return False

def script_computer(env,*args,**kwargs)->bool:
    """@."""
    return False

def script_exiting_first_town(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("start_decision")==0:
        env.force_teleport_to(0x28,3,5,1)
    env.activate_event_reward_flag("exiting_first_town")
    return True

def script_start_decision(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("exiting_first_town")>0:
        if env.get_event_flag("start_decision")==0:
            env.set_first_party_creature()
        for i in range(3):
            env.modify_npc_configs(f"pacific_town_laboratory_start_item{i+1:d}",{"state":0})
        env.activate_event_reward_flag("start_decision")
    return True

def script_professor(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("start_decision")>0 and env.get_event_flag("shop_pack")>0:
        env.modify_npc_configs("flower_city_lady",{"state":0})
        env.modify_npc_configs("flower_city_elderly_man",{"state":0})
        env.activate_event_reward_flag("encounters_tracker")
        env.drop_item_by_str("shop_pack")
    return False

def script_exit_lab_without_start_decision(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("start_decision")>0:
        return True
    if env.get_event_flag("exiting_first_town")>0:
        script_invert_direction_text(env)
    return False

def script_heal(env,*args,**kwargs)->bool:
    """@."""
    env.party_heal()
    return False

def script_heal_checkpoint(env,*args,**kwargs)->bool:
    """@."""
    env.set_checkpoint_map_position(*env.game_state["player_coordinates_data"][:4])
    script_heal(env)

def script_multiplayer_placeholder(env,*args,**kwargs)->bool:
    """@."""
    return False

def script_shop(env,*args,**kwargs)->bool:
    """@."""
    return False

def script_access_flower_city_club(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("medal1")>0 and env.get_event_flag("medal2")>0 and env.get_event_flag("medal3")>0 and env.get_event_flag("medal4")>0 and env.get_event_flag("medal5")>0 and env.get_event_flag("medal6")>0 and env.get_event_flag("medal7")>0:
        return True
    script_invert_direction_text(env)
    return False

def script_elderly_man(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("encounters_tracker")>0:
        for k in ["flower_city_lady","flower_city_elderly_man"]:
            env.modify_npc_configs(k,{"state":0})
        return True
    script_invert_direction_text(env)
    return False

def script_shop_pack(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("shop_pack")==0:
        env.activate_event_reward_flag("shop_pack")
#        env.set_item_by_str("shop_pack")
        return False
    return script_shop(env)

def script_world_map(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("encounters_tracker")>0:
#        if env.get_event_flag("world_map")==0:
#            env.set_item_by_str("world_map")
        env.activate_event_reward_flag("world_map")
    return False

def script_stone_city_tour_guy(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("medal1")>0:
        env.modify_npc_configs("stone_city_tour_guy",{"state":0})
        return True
    env.force_teleport_to(0x02,18,11,0)
    env.modify_npc_configs("stone_city_tour_guy",{"direction":0})
    return False

def script_club1_leader(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("medal1")>0 or script_npc_battle(env,*args,moves_penalty=0.75):
        env.modify_npc_configs("stone_city_tour_guy",{"state":0})
        if env.get_event_flag("medal1")==0:
            env.set_item_by_str("l_atk_01")
        env.activate_event_reward_flag("medal1")
    return False

def script_odd_stone(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("odd_stone")==0:
        for k in ["crescent_mountain_odd_stone1","crescent_mountain_odd_stone2"]:
            env.modify_npc_configs(k,{"state":0})
#        if env.get_event_flag("odd_stone")==0:
#            env.set_item_by_str("odd_stone")
        env.activate_event_reward_flag("odd_stone")
    return False

def script_odd_stone_validation(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("odd_stone")>0:
        return True
    return script_invert_direction_text(env)

def script_club2_leader(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("medal2")>0 or script_npc_battle(env,*args,moves_penalty=0.8):
        if env.get_event_flag("medal2")==0:
            env.set_item_by_str("l_atk_02")
        env.activate_event_reward_flag("medal2")
    return False

def script_bicycle_coupon(env,*args,**kwargs)->bool:
    """@."""
#    if env.get_event_flag("bicycle_coupon")==0:
#        env.set_item_by_str("bicycle_coupon")
    env.activate_event_reward_flag("bicycle_coupon")
    return False

def script_old_rod(env,*args,**kwargs)->bool:
    """@."""
#    if env.get_event_flag("old_rod")==0:
#        env.set_item_by_str("old_rod")
    env.activate_event_reward_flag("old_rod")
    return False

def script_transformed_scientist(env,*args,**kwargs)->bool:
    """@."""
    if env.get_npc_configs("sea_ranch_transformed_scientist")["x"]==4:
        env.modify_npc_configs("sea_city_guard",{"x":28})
#        if env.get_event_flag("ship_ticket")==0:
#            env.set_item_by_str("ship_ticket")
        env.activate_event_reward_flag("ship_ticket")
    else:
        env.modify_npc_configs("sea_ranch_transformed_scientist",{"y":2,"x":6,"direction":0},permanent=False)
    return False

def script_transformed_scientist_computer(env,*args,**kwargs)->bool:
    """@."""
    if env.game_state["player_coordinates_data"][3]==1 and env.get_npc_configs("sea_ranch_transformed_scientist")["y"]==2:
        env.modify_npc_configs("sea_ranch_transformed_scientist",{"y":4,"x":4,"direction":2,"sprite":22})
    return False

def script_thrashed_house_guard(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("ship_ticket")>0:
        env.modify_npc_configs("sea_city_guard",{"x":28})
        return True
    return script_invert_direction_text(env)

def script_dock_ticket_validation_a(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("ship_ticket")==0 or env.get_event_flag("powerup_debush")>0 or ((env.get_event_flag("ship_validation")==0 and env.game_state["player_coordinates_data"][0]==0x5E)):
        env.add_forced_movement_up()
    return False

def script_dock_ticket_validation_b(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("ship_validation")==0:
        env.add_forced_movement_up()
    if env.get_event_flag("powerup_debush")>0:
        script_dock_ticket_validation_a(env)
        return True
    return False

def script_ship_validation(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("ship_ticket")>0:
        env.activate_event_reward_flag("ship_validation")
    else:
        env.add_forced_movement_up()
    return False

def script_powerup_debush(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("powerup_debush")==0:
        script_club_puzzle_initialize(env)
        env.set_item_by_str("powerup_debush")
    env.activate_event_reward_flag("powerup_debush")
    return False

def script_club_puzzle_initialize(env,*args,**kwargs)->bool:
    """@."""
    if env.invincible:
        env.puzzle_random_idxs=np.array([12,11],dtype=np.int8)
    else:
        env.puzzle_random_idxs=np.random.choice(np.arange(15,dtype=np.int8),2,replace=False)
    return False

def script_club_puzzle_button(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("club_puzzle")>0:
        script_club_puzzle_clear_cleanup(env)
        return True
    configs=env.get_npc_configs(env.last_npc_name)
    selected_idx=sum([max(0,min(limit,(configs[k]-off)//2))*mult for k,off,limit,mult in zip(["y","x"],[7,1],[2,4],[5,1])])
    if not hasattr(env,"puzzle_random_idxs"):
        script_club_puzzle_initialize(env)
    if env.puzzle_random_idxs[0]<0:
        if env.puzzle_random_idxs[1]<0 or env.puzzle_random_idxs[1]==selected_idx:
            script_club_puzzle_clear_cleanup(env)
            return True
        script_club_puzzle_initialize(env)
    elif env.puzzle_random_idxs[0]==selected_idx:
        env.puzzle_random_idxs[0]=-1
    return False

def script_harbour_club_lock(env,on_tile:bool=True,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("medal3")>0 or env.get_event_flag("club_puzzle")>0:
        if not on_tile:
### APPLY GRAPHICAL CHANGES?
            pass
        return True
    if on_tile:
        script_invert_direction_text(env)
    return False

def script_club_puzzle_clear_cleanup(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("medal3")>0:
        return False
    env.activate_event_reward_flag("club_puzzle")
    script_club_puzzle_initialize(env)
    for y in range(1,4):
        for x in range(1,6):
            env.modify_npc_configs(f"club_puzzle_button{y:d}{x:d}",{"state":0})
    script_harbour_club_lock(env,False)
    return False

def script_club3_leader(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("medal3")>0 or script_npc_battle(env,*args,moves_penalty=0.8):
        script_club_puzzle_clear_cleanup(env)
        if env.get_event_flag("medal3")==0:
            env.set_item_by_str("l_atk_03")
        env.activate_event_reward_flag("medal3")
    return False

def script_powerup_torch(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("powerup_torch")==0 and env.check_creatures_owned(10):
        env.set_item_by_str("powerup_torch")
        env.activate_event_reward_flag("powerup_torch")
    return False

def script_bicycle_shop(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("bicycle_coupon")>0:
#        if env.get_event_flag("bicycle")==0:
#            env.set_item_by_str("bicycle")
        env.activate_event_reward_flag("bicycle")
    return False

def script_earth_cave_torch_validation(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("can_torch")>0:
        return True
    return script_invert_direction_text(env)

def script_club4_leader(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("medal4")>0 or script_npc_battle(env,*args,moves_penalty=0.8):
        if env.get_event_flag("medal4")==0:
            env.set_item_by_str("l_atk_04")
        env.activate_event_reward_flag("medal4")
    return False

def script_commercial_city_shop_elevator(env,*args,**kwargs)->bool:
    """@."""
    map_id=0x88 if env.game_state["previous_map_id"] in {0x7A,0x7B} else 0x7A
    env.reindex_current_map_warps([[i,map_id,1,1] for i in range(2)])
    return False

def script_gaming_house_recruit(env,*args,**kwargs)->bool:
    """@."""
    if script_npc_battle(env,*args):
        env.modify_npc_configs("gaming_house_recruit",{"state":0})
        return True
    return False

def script_unlock_evils_refuge(env,*args,**kwargs)->bool:
    """@."""
    env.activate_event_reward_flag("medal3")
    env.activate_event_reward_flag("unlock_evils_refuge")
    return False

def script_access_evils_refuge(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("unlock_evils_refuge")>0:
        return True
    return script_invert_direction_text(env)

def script_evils_refuge_recruit_elevator_key(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("elevator_key")==0:
        if env.get_npc_configs(env.last_npc_name)["won"]==0:
            script_npc_battle(env,*args)
        else:
            env.modify_npc_configs("evils_refuge_elevator_key",{"state":1})
    return False

def script_elevator_key(env,*args,**kwargs)->bool:
    """@."""
    env.modify_npc_configs("evils_refuge_elevator_key",{"state":0})
#    if env.get_event_flag("elevator_key")==0:
#        env.set_item_by_str("elevator_key")
    env.activate_event_reward_flag("elevator_key")
    return False

def script_evils_refuge_elevator(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("elevator_key")>0:
        if env.game_state["previous_map_id"]==0xCA:
            warps_data=[[i,0xC7,19,24+i] for i in range(2)]
        else:
            warps_data=[[i,0xCA,15,24+i] for i in range(2)]
        env.reindex_current_map_warps(warps_data)
    return False

def script_evils_refuge_recruit_lock_by_idx(env,idx:int=0,*args,**kwargs)->bool:
    """@."""
    if not env.using_npcs:
        return True
    required_recruits=[env.get_npc_configs(k)["won"]>0 for k in ["evils_refuge_recruit_lock1","evils_refuge_recruit_lock2"]]
    if not required_recruits[idx]:
        if script_npc_battle(env,*args):
            required_recruits[idx]=True
    if required_recruits[0] and required_recruits[1]:
        script_evils_refuge_lock(env,on_tile=False)
    return False

def script_evils_refuge_recruit_lock1(env,*args,**kwargs)->bool:
    """@."""
    return script_evils_refuge_recruit_lock_by_idx(env,0,*args)

def script_evils_refuge_recruit_lock2(env,*args,**kwargs)->bool:
    """@."""
    return script_evils_refuge_recruit_lock_by_idx(env,1,*args)

def script_evils_refuge_lock(env,on_tile:bool=True,*args,**kwargs)->bool:
    """@."""
    if not env.using_npcs:
        return True
    required_recruits=[env.get_npc_configs(k)["won"]>0 for k in ["evils_refuge_recruit_lock1","evils_refuge_recruit_lock2"]]
    if required_recruits[0] and required_recruits[1]:
        if not on_tile:
### APPLY GRAPHICAL CHANGES?
            pass
        return True
    if on_tile:
        script_invert_direction_text(env)
    return False

def script_evils_refuge_clear_cleanup(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("ghost_radar")>0:
        return False
    for k in ["evils_refuge_recruit_lock1","evils_refuge_recruit_lock2","evils_refuge_recruit_elevator_key"]:
        if env.get_npc_configs(k)["won"]==0:
            env.modify_npc_configs(k,{"won":1,"interactions":1})
    return False

def script_evils_refuge_admin(env,*args,**kwargs)->bool:
    """@."""
    if env.get_npc_configs("evils_refuge_admin")["won"]>0:
        return False
    if env.get_event_flag("ghost_radar")>0 or script_npc_battle(env,*args,moves_penalty=0.8):
        env.modify_npc_configs("evils_refuge_admin",{"state":0})
        if env.get_event_flag("ghost_radar")==0:
            env.modify_npc_configs("evils_refuge_ghost_radar",{"state":1})
        script_evils_refuge_clear_cleanup(env)
    return False

def script_ghost_radar(env,*args,**kwargs)->bool:
    """@."""
    env.modify_npc_configs("evils_refuge_ghost_radar",{"state":0})
#    if env.get_event_flag("ghost_radar")==0:
#        env.set_item_by_str("ghost_radar")
    env.activate_event_reward_flag("ghost_radar")
    return False

def script_ghost_doll(env,*args,**kwargs)->bool:
    """@."""
#    if env.get_event_flag("ghost_doll")==0:
#        env.set_item_by_str("ghost_doll")
    env.activate_event_reward_flag("ghost_doll")
    return False

def script_water_bottle(env,*args,**kwargs)->bool:
    """@."""
#    if env.get_event_flag("water_bottle")==0:
#        env.set_item_by_str("water_bottle")
    env.activate_event_reward_flag("water_bottle")
    return False

def script_powerup_teleport(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("powerup_teleport")==0:
        env.set_item_by_str("powerup_teleport")
    env.activate_event_reward_flag("powerup_teleport")
    return False

def script_ghost_enemy(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("ghost_enemy")>0 or (env.get_event_flag("ghost_radar")>0 or env.get_event_flag("ghost_doll")>0):
        if env.get_event_flag("ghost_enemy")>0 or env.headless_battle_npc(level=30,num=1,moves_penalty=0.75):
            env.modify_npc_configs("creepy_tower_ghost_enemy",{"state":0})
            env.activate_event_reward_flag("ghost_enemy")
            return True
        return False
    return script_invert_direction_text(env)

def script_save_kidnapped_elderly(env,*args,**kwargs)->bool:
    """@."""
    env.modify_npc_configs("creepy_tower_kidnapped_elderly",{"state":0})
    env.modify_npc_configs("creepy_town_kidnapped_elderly",{"state":1})
    env.modify_npc_configs("corporate_city_recruit_block_building",{"x":19})
    env.activate_event_reward_flag("save_kidnapped_elderly")
    script_teleport_to_kidnapped_elderly_house(env)
    return True

def script_teleport_to_kidnapped_elderly_house(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("save_kidnapped_elderly")>0:
        env.force_teleport_to(0x95,7,3,1)
        return True
    return False

def script_megaphone(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("megaphone")==0:
        env.set_item_by_str("megaphone")
    env.activate_event_reward_flag("megaphone")
    return False

def script_best_rod(env,*args,**kwargs)->bool:
    """@."""
#    if env.get_event_flag("best_rod")==0:
#        env.set_item_by_str("best_rod")
    env.activate_event_reward_flag("best_rod")
    return False

def script_awaken_sleeping_creature_by_idx(env,idx:int,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("megaphone")>0:
        if env.get_event_flag("awaken_sleeping_creature")>0 or env.headless_battle_npc(level=30,num=1,moves_penalty=0.75):
            env.modify_npc_configs(f"sleeping_creature{idx:d}",{"state":0})
            env.activate_event_reward_flag("awaken_sleeping_creature")
            return True
    script_invert_direction_text(env)
    return False

def script_awaken_sleeping_creature1(env,*args,**kwargs)->bool:
    """@."""
    return script_awaken_sleeping_creature_by_idx(env,1)

def script_awaken_sleeping_creature2(env,*args,**kwargs)->bool:
    """@."""
    return script_awaken_sleeping_creature_by_idx(env,2)

def script_gate_guard(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("water_bottle")>0:
        return True
    return script_invert_direction_text(env)

def script_cycling_area_guard(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("bicycle")>0:
        return True
    return script_invert_direction_text(env)

def script_club5_leader(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("medal5")>0 or script_npc_battle(env,*args,moves_penalty=0.8):
        if env.get_event_flag("medal5")==0:
            env.set_item_by_str("l_atk_05")
        env.activate_event_reward_flag("medal5")
    return False

def script_medium_rod(env,*args,**kwargs)->bool:
    """@."""
#    if env.get_event_flag("medium_rod")==0:
#        env.set_item_by_str("medium_rod")
    env.activate_event_reward_flag("medium_rod")
    return False

def script_trek_start(env,force:bool=False):
    """@."""
    if force or env.cached_is_trek_map(env.game_state["player_coordinates_data"][0]):
        env.game_state["trek_timeout"]=500# if not env.invincible and env.action_complexity>=3 else 32000
    return False

def script_trek_timer(env,*args,**kwargs)->bool:
    """@."""
    if env.cached_is_trek_map(env.game_state["player_coordinates_data"][0]):
        env.game_state["trek_timeout"]-=1
        if env.game_state["trek_timeout"]<1 or env.get_event_flag("powerup_swim")>0:
            return script_trek_end(env)
    return False

def script_trek_end(env,force:bool=False):
    """@."""
    if force or env.cached_is_trek_map(env.game_state["player_coordinates_data"][0]):
        env.game_state["trek_timeout"]=0
        env.force_teleport_to(0x9C,0,4,0)
    return False

def script_dentures(env,*args,**kwargs)->bool:
    """@."""
    env.modify_npc_configs("trek_area_west_dentures",{"state":0})
#    if env.get_event_flag("dentures")==0:
#        env.set_item_by_str("dentures")
    env.activate_event_reward_flag("dentures")
    return False

def script_powerup_swim(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("powerup_swim")==0:
        env.set_item_by_str("powerup_swim")
    env.activate_event_reward_flag("powerup_swim")
    return False

def script_powerup_move_boulders(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("dentures")>0:
        if env.get_event_flag("powerup_move_boulders")==0:
            env.set_item_by_str("powerup_move_boulders")
        env.activate_event_reward_flag("powerup_move_boulders")
    return False

def script_fighting_club_prize(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("fighting_club_prize")==0:
        if not env.using_npcs or env.get_npc_configs("corporate_city_fighting_club_leader")["won"]>0:
            for k in ["corporate_city_fighting_club_prize1","corporate_city_fighting_club_prize2"]:
                env.modify_npc_configs(k,{"state":0})
#            if env.get_event_flag("fighting_club_prize")==0:
#                 env.set_item_by_str("fighting_club_prize")
            env.activate_event_reward_flag("fighting_club_prize")
    return False

def script_corporate_recruit_block_building(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("save_corporate_president")>0:
        env.modify_npc_configs("corporate_city_recruit_block_building",{"state":0})
        script_corporate_building_initialize(env)
        return True
    if env.get_event_flag("megaphone")>0:
        env.modify_npc_configs("corporate_city_recruit_block_building",{"x":19})
        return True
    return script_invert_direction_text(env)

def script_corporate_recruit_block_by_name(env,name:str,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("save_corporate_president")>0:
        env.modify_npc_configs(name,{"state":0})
        return True
    return script_invert_direction_text(env)

def script_corporate_recruit_block_club(env,*args,**kwargs)->bool:
    """@."""
    return script_corporate_recruit_block_by_name(env,"corporate_city_recruit_block_club")

def script_corporate_recruit_block_house1(env,*args,**kwargs)->bool:
    """@."""
    return script_corporate_recruit_block_by_name(env,"corporate_city_recruit_block_house1")

def script_corporate_recruit_block_house2(env,*args,**kwargs)->bool:
    """@."""
    return script_corporate_recruit_block_by_name(env,"corporate_city_recruit_block_house2")

def script_corporate_building_clear_cleanup(env,*args,**kwargs)->bool:
    """@."""
    for k in ["saffron_city_recruit_block_corporate_building","saffron_city_recruit_block_club","saffron_city_recruit_block_house1","saffron_city_recruit_block_house2"]:
        env.modify_npc_configs(k,{"state":0})
    return False

def script_corporate_building_entrance(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("water_bottle")>0 and env.get_event_flag("megaphone")>0:
        script_corporate_building_initialize(env)
        return True
    return script_invert_direction_text(env)

def script_corporate_building_initialize(env,*args,**kwargs)->bool:
    """@."""
    return False

def script_corporate_building_elevator(env,*args,**kwargs)->bool:
    """@."""
    if env.game_state["previous_map_id"] in {0xB5,0xCF,0xD0,0xD1,0xD2,0xD3}:
        warps_data=[[i,0xE9,0,18] for i in range(2)]
    else:
        warps_data=[[i,0xB5,0,20] for i in range(2)]
    env.reindex_current_map_warps(warps_data)
    return False

def script_corporate_pass(env,*args,**kwargs)->bool:
    """@."""
    env.modify_npc_configs("corporate_building_pass",{"state":0})
#    if env.get_event_flag("corporate_pass")==0:
#        env.set_item_by_str("corporate_pass")
    env.activate_event_reward_flag("corporate_pass")
    return False

def script_corporate_building_lock(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("corporate_pass")>0:
        return True
    return script_invert_direction_text(env)

def script_corporate_building_admin(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("corporate_pass")==0 or env.get_npc_configs("corporate_building_admin")["won"]>0:
        return False
    if env.get_event_flag("save_corporate_president")>0 or script_npc_battle(env,*args,moves_penalty=0.8):
        env.modify_npc_configs("corporate_building_admin",{"state":0})
    return False

def script_save_corporate_president(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("corporate_pass")==0 or env.get_npc_configs("corporate_building_admin")["won"]<1:
        return False
#    if env.get_event_flag("save_corporate_president")==0:
#        env.set_item_by_str("corporate_product")
    env.activate_event_reward_flag("save_corporate_president")
    script_corporate_building_clear_cleanup(env)
    return False

def script_club6_leader(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("medal6")>0 or script_npc_battle(env,*args,moves_penalty=0.8):
        if env.get_event_flag("medal6")==0:
            env.set_item_by_str("l_atk_06")
        env.activate_event_reward_flag("medal6")
    return False

def script_vulcanic_refuge_doors_switches_initialize(env,*args,**kwargs)->bool:
    """@."""
    return False

def script_mansion_doors_switches_toggle(env,*args,**kwargs)->bool:
    """@."""
    return False

def script_mansion_fall_a(env,*args,**kwargs)->bool:
    """@."""
    env.force_teleport_to(0xA5,14,16,0)
    return False

def script_mansion_fall_b(env,*args,**kwargs)->bool:
    """@."""
    env.force_teleport_to(0xD6,14,18,0)
    return False

def script_vulcanic_club_key(env,*args,**kwargs)->bool:
    """@."""
    env.modify_npc_configs("vulcanic_club_key",{"state":0})
#    if env.get_event_flag("vulcanic_club_key")==0:
#        env.set_item_by_str("vulcanic_club_key")
    env.activate_event_reward_flag("vulcanic_club_key")
    return False

def script_access_vulcanic_city_club(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("vulcanic_club_key")>0:
        return True
    return script_invert_direction_text(env)

def script_club7_leader(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("medal7")>0 or script_npc_battle(env,*args,moves_penalty=0.85):
        if env.get_event_flag("medal7")==0:
            env.set_item_by_str("l_atk_07")
        env.activate_event_reward_flag("medal7")
    return False

def script_club8_leader(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("medal8")>0 or script_npc_battle(env,*args,moves_penalty=0.85):
        if env.get_event_flag("medal8")==0:
            env.set_item_by_str("l_atk_08")
        env.activate_event_reward_flag("medal8")
    return False

def script_guard_all_medals_check(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("medal8")==0:
        script_invert_direction_text(env)
    return False

def script_champions_cave_move_boulders_validation(env,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("can_move_boulders")>0:
        return True
    return script_invert_direction_text(env)

def script_champions_cave_lock1(env,*args,**kwargs)->bool:
    """@."""
    return False

def script_champions_highland_master_validation_by_name(env,name:str,*args,**kwargs)->bool:
    """@."""
    if len(name)==0 or not env.using_npcs or env.get_npc_configs(name)["won"]>0:
        env.add_forced_movement_up()
    else:
        script_invert_direction_text(env)
    return False

def script_champions_highland_master0_validation(env,*args,**kwargs)->bool:
    """@."""
    return script_champions_highland_master_validation_by_name(env,"")

def script_champions_highland_master1_validation(env,*args,**kwargs)->bool:
    """@."""
    return script_champions_highland_master_validation_by_name(env,"champions_highland_master1")

def script_champions_highland_master2_validation(env,*args,**kwargs)->bool:
    """@."""
    return script_champions_highland_master_validation_by_name(env,"champions_highland_master2")

def script_champions_highland_master3_validation(env,*args,**kwargs)->bool:
    """@."""
    return script_champions_highland_master_validation_by_name(env,"champions_highland_master3")

def script_champions_highland_master4_validation(env,*args,**kwargs)->bool:
    """@."""
    return script_champions_highland_master_validation_by_name(env,"champions_highland_master4")

def script_champions_highland_champion_validation(env,*args,**kwargs)->bool:
    """@."""
    return script_champions_highland_master_validation_by_name(env,"champions_highland_champion")


def script_champions_highland_master_by_name(env,name:str,*args,**kwargs)->bool:
    """@."""
    if not script_npc_battle(env,*args,moves_penalty=0.9):
        script_champions_highland_cleanup(env)
    return False

def script_champions_highland_cleanup(env,*args,**kwargs)->bool:
    """@."""
    for k in [f"champions_highland_master{i:d}" for i in range(1,5)]+["champions_highland_champion"]:
        env.modify_npc_configs(k,{"won":0})

def script_champions_highland_master1(env,*args,**kwargs)->bool:
    """@."""
    return script_champions_highland_master_by_name(env,"champions_highland_master1",*args,**kwargs)

def script_champions_highland_master2(env,*args,**kwargs)->bool:
    """@."""
    return script_champions_highland_master_by_name(env,"champions_highland_master2",*args,**kwargs)

def script_champions_highland_master3(env,*args,**kwargs)->bool:
    """@."""
    return script_champions_highland_master_by_name(env,"champions_highland_master3",*args,**kwargs)

def script_champions_highland_master4(env,*args,**kwargs)->bool:
    """@."""
    return script_champions_highland_master_by_name(env,"champions_highland_master4",*args,**kwargs)

def script_champions_highland_champion(env,*args,**kwargs)->bool:
    """@."""
    return script_champions_highland_master_by_name(env,"champions_highland_champion",*args,**kwargs)

def script_victory_room(env,*args,**kwargs)->bool:
    """@."""
    script_champions_highland_cleanup(env)
    env.activate_event_reward_flag("victory_room")
    env.set_checkpoint_map_position(*env.first_checkpoint_place)
    env.force_teleport_to(0x00,6,5,0)

def script_secret_creature_bird_e(env,idx:int,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("powerup_swim")>0 and env.get_event_flag("medal5")>0:
        if env.get_event_flag("secret_creature_bird_e")>0 or env.headless_battle_npc(level=50,num=1,moves_penalty=0.9):
            env.modify_npc_configs("secret_creature_bird_e",{"state":0})
            env.activate_event_reward_flag("secret_creature_bird_e")
            return True
    script_invert_direction_text(env)
    return False

def script_secret_creature_bird_i(env,idx:int,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("powerup_swim")>0 and env.get_event_flag("medal5")>0 and env.get_event_flag("powerup_move_boulders")>0 and env.get_event_flag("medal4")>0:
        if env.get_event_flag("secret_creature_bird_i")>0 or env.headless_battle_npc(level=50,num=1,moves_penalty=0.9):
            env.modify_npc_configs("secret_creature_bird_i",{"state":0})
            env.activate_event_reward_flag("secret_creature_bird_i")
            return True
    script_invert_direction_text(env)
    return False

def script_secret_creature_bird_f(env,idx:int,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("powerup_move_boulders")>0 and env.get_event_flag("all_medals_check")>0:
        if env.get_event_flag("secret_creature_bird_f")>0 or env.headless_battle_npc(level=50,num=1,moves_penalty=0.9):
            env.modify_npc_configs("secret_creature_bird_f",{"state":0})
            env.activate_event_reward_flag("secret_creature_bird_f")
            return True
    script_invert_direction_text(env)
    return False

def script_secret_creature_monster_p(env,idx:int,*args,**kwargs)->bool:
    """@."""
    if env.get_event_flag("victory_room")>0:
        if env.get_event_flag("secret_creature_monster_p")>0 or env.headless_battle_npc(level=70,num=1,moves_penalty=0.95):
            env.modify_npc_configs("secret_creature_monster_p",{"state":0})
            env.activate_event_reward_flag("secret_creature_monster_p")
            return True
    script_invert_direction_text(env)
    return False
