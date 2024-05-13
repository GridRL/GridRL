#!/usr/bin/env python3

"""Core functions of the game exploration_world1."""

from typing import Union,Any
import sys
import os
import numpy as np
sys.dont_write_bytecode=True

if __package__ is None or len(__package__)==0:
    try:
        __file__=__file__
    except NameError:
        __file__=""
    sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}{os.sep}..{os.sep}..")
#    sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}{os.sep}..{os.sep}..{os.sep}abstract_games")
try:
    from gridrl.core_constants import (meadow_tile_id,water_tile_id,
        ground_tile_id,snow_tile_id,
        bush_tile_id,whirlpool_tile_id,
        waterfall_tile_id,breakable_rock_tile_id,mountain_climb_tile_id,
        breakable_frozen_rock_tile_id)
    from gridrl.abstract_games.exploration_abstract_game import ExplorationAbstractGame
    from gridrl.games.exploration_world1.menu import ExplorationWorld1Menu
    from gridrl.games.exploration_world1.constants import powerups_list
except ModuleNotFoundError:
    from core_constants import (meadow_tile_id,water_tile_id,
        ground_tile_id,snow_tile_id,
        bush_tile_id,whirlpool_tile_id,
        waterfall_tile_id,breakable_rock_tile_id,mountain_climb_tile_id,
        breakable_frozen_rock_tile_id)
    from abstract_games.exploration_abstract_game import ExplorationAbstractGame
    from games.exploration_world1.menu import ExplorationWorld1Menu
    from games.exploration_world1.constants import powerups_list

__all__=["ExplorationWorld1Game"]

class ExplorationWorld1Game(ExplorationAbstractGame):
    """The main implementation of the game exploration_world1."""
    def __init__(self,
        config:Union[dict,None]=None,
        agent_class:Union[Any,None]=None,
        agent_args:Union[dict,None]=None,
        *args,**kwargs
    )->None:
        """Constructor."""
        game_name="exploration_world1"
###############################
### DEFINING NEW ATTRIBUTES ###
### ALSO DEFINE THEM VIA    ###
### METHODS LISTED BELOW    ###
###############################
        self.sandbox=False
        self.action_debush_id=0
        self.action_swim_id=0
        self.action_teleport_id=0
        self.action_cross_whirlpool_id=0
        self.action_cross_waterfall_id=0
        self.action_break_rock_id=0
        self.action_mountain_climb_id=0
        self.action_break_frozen_rock_id=0
        self.powerups_ids=np.array([i for i,k in enumerate(powerups_list)],dtype=np.uint8,order="C")
        self.powerups_action_ids={}
##################
### SUPER INIT ###
##################
        super().__init__(game_name=game_name,config=config,
            agent_class=agent_class,agent_args=agent_args,*args,**kwargs)
#################################
### GAME SPECIFIC DEFINITIONS ###
#################################
    def get_menu_class(self)->Any:
        """Return the class of the menu object."""
        return ExplorationWorld1Menu
    def game_enforce_config(self,config:dict)->dict:
        """Alter the configurations dict to enforce specific options."""
        config["screen_observation_type"]=max(0,min(3,int(config.get("screen_observation_type",-1))))
#        config["action_complexity"]=max(2,min(4,int(config.get("action_complexity",-1))))
        config["strip_ledges"]=False
        config["action_nop"]=False
        return config
    def define_game_tile_size(self)->int:
        """Sprite/Tile size in pixels."""
        return 16
    def define_game_screen_tiles_size(self)->tuple:
        """Screen size in tiles unit."""
        return (9,10)
    def define_game_extra_data_functions(self)->list:
        """List of extra game data files functions."""
        return []
    def define_extra_required_game_data_names(self)->list:
        """List of required keys that must exist in the dictionary data."""
        return []
    def define_game_config(self,config:dict)->None:
        """Game-specific configurations initialization."""
        super().define_game_config(config)
        if self.sandbox:
            sandbox_start_pos=[0x1C,6,5,0]
            self.set_start_positions(start=sandbox_start_pos,checkpoint=sandbox_start_pos)
            self.starting_event=""
            self.starting_collected_flags=["exiting_first_town"]
            self.starting_collected_flags+=[f"medal{i:d}" for i in range(1,9)]
            self.starting_collected_flags+=[f"powerup_{k[4:]}" for k,_ in self.get_game_powerup_tiles_dict().items()]
            self.starting_collected_flags+=powerups_list
    def define_internal_data(self)->None:
        """Game-specific attribute declarations."""
        super().define_internal_data()
    def define_actions_ids(self)->int:
        """Custom game actions id declaration."""
        super().define_actions_ids()
        base_action_id=self.action_menu_max_id if self.true_menu else self.action_interact_id
        self.action_debush_id=base_action_id+1
        self.action_swim_id=base_action_id+2
        self.action_teleport_id=base_action_id+3
        self.action_cross_whirlpool_id=base_action_id+4
        self.action_cross_waterfall_id=base_action_id+5
        self.action_break_rock_id=base_action_id+6
        self.action_mountain_climb_id=base_action_id+7
        self.action_break_frozen_rock_id=base_action_id+8
        self.powerups_action_ids={
            "powerup_debush":self.action_debush_id,
            "powerup_swim":self.action_swim_id,
            "powerup_teleport":0,#self.action_teleport_id,
            "powerup_cross_whirlpool":self.action_cross_whirlpool_id,
            "powerup_cross_waterfall":self.action_cross_waterfall_id,
            "powerup_break_rock":self.action_break_rock_id,
            "powerup_mountain_climb":self.action_mountain_climb_id,
            "powerup_break_frozen_rock":self.action_break_frozen_rock_id,
        }
        if not self.bypass_powerup_actions:
            return 5#8
        return 0
    def define_extra_game_state(self)->dict:
        """Dict of extra values preserved in the game_state dictionary."""
        state=super().define_extra_game_state()
        state.update({"money":0,"party_size":0,"bag_size":0,
            "encounter_steps":0,
            "bag":np.zeros((1,2),np.uint8,order="C"),
            "party_levels":np.zeros((1,),np.uint8,order="C"),
            "party_hp_ratios":np.zeros((1,),np.float32,order="C"),
        })
        return state
    def define_extra_stacked_state(self)->dict:
        """Dict of extra stacked values preserved in the stacked_state dictionary."""
        return {}
#############
### TILES ###
#############
    def get_game_powerup_tiles_dict(self)->dict:
        """Return a dict with powerup keys and walkable tiles list as values."""
        return {"can_debush":[bush_tile_id],
            "can_swim":[water_tile_id],
            "can_cross_whirlpool":[whirlpool_tile_id],
            "can_cross_waterfall":[waterfall_tile_id],
            "can_break_rock":[breakable_rock_tile_id],
            "can_mountain_climb":[mountain_climb_tile_id],
            "can_break_frozen_rock":[breakable_frozen_rock_tile_id],
        }
######################
### EVENT-HANDLING ###
######################
    def define_game_critical_event_names(self)->None:
        """Define first and last events of the game."""
        self.first_event_name="exiting_first_town"
        self.trigger_done_event_name="engineer_guard"
    def game_on_event_custom_load(self,starting_event:str,used_collected_flags:set,used_level:int)->None:
        """Game-specific custom state load fixes."""
        return
    def game_on_event_flag_change(self,event_name:str,activated:bool)->bool:
        """Used to for validating event states. Return True to clear again the event cache."""
        return False
##################
### DEBUG TEXT ###
##################
    def get_party_text(self)->str:
        """Party summary text."""
        return f"Level: {self.game_state['party_levels'][0]:3d}\tHP: {self.game_state['party_hp_ratios'][0]:.0%}"
    def get_game_debug_text(self)->str:
        """Game speficic debug text."""
        return f"Party\t{self.get_party_text()}"
    def get_game_commands_text(self)->str:
        """Game speficic buttons input text."""
        return ""
##################
### DATA NAMES ###
##################
    def get_game_attribute_state_names(self)->list[str]:
        """List of game-specific attribute names preserved in a save state."""
        return []
###########################
### STRUCTURE FUNCTIONS ###
###########################
    def get_powerup_id(self,idx:int)->int:
        """Return the powerup id."""
        return idx
    def use_powerup(self,powerup_id:int)->bool:
        """Binds usage of the field move withing the environment."""
        if powerup_id in self.powerups_ids:
            action_id=self.powerups_action_ids.get(powerups_list[powerup_id],0)
            if action_id>0:
                self.add_forced_action(action_id)
                return True
        return False
###############################
### STRUCTURES CONDITIONALS ###
###############################
    def can_use_powerup(self,powerup_id)->None:
        """Is allowed to use the powerup."""
        return self.get_event_flag(f"can_{powerups_list[powerup_id][8:]}")>0
##################
### GAME LOGIC ###
##################
    def handle_non_directional_actions(self,action:int)->tuple:
        """Game-specific logic for non-movement buttons."""
        (skip_movement,powerup_started)=(True,False)
        self.game_state["powerup_screen_remove_tile"]=0
        if action==self.action_debush_id:
            if self.get_event_flag("can_debush")>0:
                if self.game_state["powerup_walk_tile"]==0:
                    self.game_state["powerup_walk_tile"]=bush_tile_id
                    self.game_state["powerup_screen_remove_tile"]=bush_tile_id
                    self.game_state["powerup_screen_fix_tile"]=meadow_tile_id
        elif action==self.action_swim_id:
            if self.get_event_flag("can_swim")>0:
                if self.game_state["powerup_walk_tile"]!=water_tile_id!=self.game_data["maps"][self.game_state["player_coordinates_data"][0]][self.game_state["player_coordinates_data"][1],self.game_state["player_coordinates_data"][2]] and self.get_faced_tile()==water_tile_id:
                    (skip_movement,powerup_started,self.game_state["powerup_walk_tile"])=(False,True,water_tile_id)
        elif action==self.action_teleport_id:
            pass
        elif action==self.action_cross_whirlpool_id:
            if self.get_event_flag("can_cross_whirlpool")>0:
                if self.game_state["powerup_walk_tile"]==water_tile_id:
                    self.game_state["powerup_walk_tile"]=whirlpool_tile_id
                    self.game_state["powerup_screen_remove_tile"]=whirlpool_tile_id
                    self.game_state["powerup_screen_fix_tile"]=water_tile_id
        elif action==self.action_cross_waterfall_id:
            if self.get_event_flag("can_cross_waterfall")>0:
                if self.game_state["powerup_walk_tile"]!=waterfall_tile_id!=self.game_data["maps"][self.game_state["player_coordinates_data"][0]][self.game_state["player_coordinates_data"][1],self.game_state["player_coordinates_data"][2]] and self.get_faced_tile()==waterfall_tile_id:
                    (skip_movement,powerup_started,self.game_state["powerup_walk_tile"])=(False,True,waterfall_tile_id)
        elif action==self.action_break_rock_id:
            if self.get_event_flag("can_break_rock")>0:
                if self.game_state["powerup_walk_tile"]==0:
                    self.game_state["powerup_walk_tile"]=breakable_rock_tile_id
                    self.game_state["powerup_screen_remove_tile"]=breakable_rock_tile_id
                    self.game_state["powerup_screen_fix_tile"]=ground_tile_id
        elif action==self.action_mountain_climb_id:
            if self.get_event_flag("can_mountain_climb")>0:
                if self.game_state["powerup_walk_tile"]!=mountain_climb_tile_id!=self.game_data["maps"][self.game_state["player_coordinates_data"][0]][self.game_state["player_coordinates_data"][1],self.game_state["player_coordinates_data"][2]] and self.get_faced_tile()==mountain_climb_tile_id:
                    (skip_movement,powerup_started,self.game_state["powerup_walk_tile"])=(False,True,mountain_climb_tile_id)
        elif action==self.action_break_frozen_rock_id:
            if self.get_event_flag("can_break_frozen_rock")>0:
                if self.game_state["powerup_walk_tile"]==0:
                    self.game_state["powerup_walk_tile"]=breakable_frozen_rock_tile_id
                    self.game_state["powerup_screen_remove_tile"]=breakable_frozen_rock_tile_id
                    self.game_state["powerup_screen_fix_tile"]=snow_tile_id
        return (skip_movement,powerup_started)
    def game_bypass_powerup_tiles(self,tile:int)->None:
        """Automatic bypass actions of powerups if event prerequisites are met."""
        if tile==bush_tile_id:
            if self.get_event_flag("can_debush")>0:
                self.game_state["powerup_walk_tile"]=bush_tile_id
        elif tile==water_tile_id:
            if self.get_event_flag("can_swim")>0:
                self.game_state["powerup_walk_tile"]=water_tile_id
        elif tile==whirlpool_tile_id:
            if self.get_event_flag("can_cross_whirlpool")>0:
                self.game_state["powerup_walk_tile"]=whirlpool_tile_id
        elif tile==waterfall_tile_id:
            if self.get_event_flag("can_cross_whirlpool")>0:
                self.game_state["powerup_walk_tile"]=waterfall_tile_id
        elif tile==breakable_rock_tile_id:
            if self.get_event_flag("can_break_rock")>0:
                self.game_state["powerup_walk_tile"]=breakable_rock_tile_id
        elif tile==mountain_climb_tile_id:
            if self.get_event_flag("can_mountain_climb")>0:
                self.game_state["powerup_walk_tile"]=mountain_climb_tile_id
        elif tile==breakable_frozen_rock_tile_id:
            if self.get_event_flag("can_break_frozen_rock")>0:
                self.game_state["powerup_walk_tile"]=breakable_frozen_rock_tile_id
    def game_powerup_fix_tile(self,tile:int,powerup_started:bool)->int:
        """Fix the tile for powerup-logic purposes."""
        if self.game_state["powerup_walk_tile"]==water_tile_id:
            if powerup_started and not self.is_swimmable_tile_id(tile,True):
                tile=0
        elif self.game_state["powerup_walk_tile"]==whirlpool_tile_id:
            if tile==water_tile_id:
                self.game_state["powerup_walk_tile"]=water_tile_id
        elif self.game_state["powerup_walk_tile"]==waterfall_tile_id:
            if powerup_started and tile!=self.game_state["powerup_walk_tile"]:
                tile=0
            elif tile==water_tile_id:
                self.game_state["powerup_walk_tile"]=water_tile_id
        elif self.game_state["powerup_walk_tile"]==mountain_climb_tile_id:
            if powerup_started and tile!=self.game_state["powerup_walk_tile"]:
                tile=0
        return tile
    def game_powerup_first_time_use_events(self,action:int,tile:int)->None:
        """Set events for first-time powerup usage."""
        if bush_tile_id==self.game_state["powerup_walk_tile"]==tile:
            if self.get_event_flag("can_debush")>0 and self.get_event_flag("first_can_debush")==0:
                self.activate_event_flag("first_can_debush")
        elif water_tile_id==self.game_state["powerup_walk_tile"]==tile:
            if self.get_event_flag("can_swim")>0 and self.get_event_flag("first_can_swim")==0:
                self.activate_event_flag("first_can_swim")
#        elif action==self.action_id_teleport:
#            if self.game_state["player_coordinates_data"][4]==0:
#                if self.get_event_flag("can_teleport")>0 and self.get_event_flag("first_can_teleport")==0:
#                    self.activate_event_flag("first_can_teleport")
        elif whirlpool_tile_id==self.game_state["powerup_walk_tile"]==tile:
            if self.get_event_flag("can_cross_whirlpool")>0 and self.get_event_flag("first_can_cross_whirlpool")==0:
                self.activate_event_flag("first_can_cross_whirlpool")
        elif waterfall_tile_id==self.game_state["powerup_walk_tile"]==tile:
            if self.get_event_flag("can_cross_waterfall")>0 and self.get_event_flag("first_can_cross_waterfall")==0:
                self.activate_event_flag("first_can_cross_waterfall")
        elif breakable_rock_tile_id==self.game_state["powerup_walk_tile"]==tile:
            if self.get_event_flag("can_break_rock")>0 and self.get_event_flag("first_can_break_rock")==0:
                self.activate_event_flag("first_can_break_rock")
        elif mountain_climb_tile_id==self.game_state["powerup_walk_tile"]==tile:
            if self.get_event_flag("can_mountain_climb")>0 and self.get_event_flag("first_can_mountain_climb")==0:
                self.activate_event_flag("first_can_mountain_climb")
        elif breakable_frozen_rock_tile_id==self.game_state["powerup_walk_tile"]==tile:
            if self.get_event_flag("can_break_frozen_rock")>0 and self.get_event_flag("first_can_break_frozen_rock")==0:
                self.activate_event_flag("first_can_break_frozen_rock")
    def game_post_movement_powerup_status(self,tile:int)->None:
        """Finalizes the state of powerup-usage."""
        for _ in range(2):
            if self.game_state["powerup_walk_tile"] in [bush_tile_id,breakable_rock_tile_id,breakable_frozen_rock_tile_id]:
                self.game_state["powerup_walk_tile"]=0
                self.game_state["powerup_screen_remove_tile"]=0
                break
            if water_tile_id==self.game_state["powerup_walk_tile"]:
                if not self.is_swimmable_tile_id(tile,True) and self.game_state["powerup_walk_tile"]!=whirlpool_tile_id:
                    self.game_state["powerup_walk_tile"]=0
                break
            if self.game_state["powerup_walk_tile"]==whirlpool_tile_id:
                self.game_state["powerup_walk_tile"]=water_tile_id
                self.game_state["powerup_screen_remove_tile"]=0
            elif waterfall_tile_id==self.game_state["powerup_walk_tile"]:
                if self.game_state["powerup_walk_tile"]!=waterfall_tile_id:
                    self.game_state["powerup_walk_tile"]=water_tile_id if self.is_swimmable_tile_id(tile,True) else 0
                break
            elif mountain_climb_tile_id==self.game_state["powerup_walk_tile"]:
                if mountain_climb_tile_id!=tile:
                    self.game_state["powerup_walk_tile"]=0
                break
        if self.game_state["powerup_walk_tile"]==0:
            self.game_state["powerup_screen_remove_tile"]=0
            self.game_state["powerup_screen_fix_tile"]=0
##################
### RESET-GAME ###
##################
    def reset_party(self)->None:
        """Initialize the party data."""
        for k in ["party_levels","party_hp_ratios"]:
            self.game_state[k][:]=0
        self.game_state["party_size"]=0
    def game_on_reset(self)->None:
        """Game-specific reset to initial state."""
        self.reset_party()
        for k in ["money","encounter_steps"]:
            self.game_state[k]=0
