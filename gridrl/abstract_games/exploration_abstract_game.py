#!/usr/bin/env python3

"""Declaration of exploration_abstract_game."""

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
    sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}{os.sep}..")
#    from functions_numba import nb_in_array_uint8
    from core_game import (GameCore,lru_cache_func)
    from core_constants import water_tile_id,bush_tile_id,swimmable_tiles_ids
else:
    try:
#        from gridrl.functions_numba import nb_in_array_uint8
        from gridrl.core_game import GameCore,lru_cache_func
        from gridrl.core_constants import water_tile_id,bush_tile_id,swimmable_tiles_ids
    except ModuleNotFoundError:
#        from functions_numba import nb_in_array_uint8
        from core_game import GameCore,lru_cache_func
        from core_constants import water_tile_id,bush_tile_id,swimmable_tiles_ids

__all__=["ExplorationAbstractGame"]

class ExplorationAbstractGame(GameCore):
    """The main implementation of a exploration_abstract_game."""
    def __init__(self,
        game_name:str,
        config:Union[dict,None]=None,
        agent_class:Union[Any,None]=None,
        agent_args:Union[dict,None]=None,
        *args,**kwargs
    )->None:
        """Constructor."""
###############################
### DEFINING NEW ATTRIBUTES ###
### ALSO DEFINE THEM VIA    ###
### METHODS LISTED BELOW    ###
###############################
        self.swimmable_tiles=np.array(swimmable_tiles_ids,dtype=np.uint8)
        self.action_debush_id=0
        self.action_swim_id=0
        self.action_teleport_id=0
##################
### SUPER INIT ###
##################
        super().__init__(game_name=game_name,config=config,
            agent_class=agent_class,agent_args=agent_args,*args,**kwargs)
#################################
### GAME SPECIFIC DEFINITIONS ###
#################################
    def game_enforce_config(self,config:dict)->dict:
        """Alter the configurations dict to enforce specific options."""
        return {}
    def define_game_tile_size(self)->int:
        """Sprite/Tile size in pixels."""
        return 16
    def define_internal_data(self)->None:
        """Game-specific attribute declarations."""
        self.swimmable_tiles=np.array(swimmable_tiles_ids,dtype=np.uint8)
    def define_actions_ids(self)->int:
        """Custom game actions id declaration."""
        base_action_id=self.action_menu_max_id if self.true_menu else self.action_interact_id
        self.action_debush_id=base_action_id+1
        self.action_swim_id=base_action_id+2
        self.action_teleport_id=base_action_id+3
        if not self.bypass_powerup_actions:
            return 2
        return 0
    def define_extra_game_state(self)->dict:
        """Dict of extra values preserved in the game_state dictionary."""
        return {"money":0,"party_size":0,
            "party_levels":np.zeros((1,),np.uint8,order="C"),
            "party_hp_ratios":np.zeros((1,),np.float32,order="C"),
        }
#############
### TILES ###
#############
    @lru_cache_func(maxsize=512)
    def is_swimmable_tile_id(self,tile_id:int,in_water:bool=False)->bool:
        """Return True if the tile can be traversed with powerup_swim."""
        return tile_id in self.swimmable_tiles if in_water else tile_id==water_tile_id
#        return nb_in_array_uint8(tile_id,self.swimmable_tiles) if in_water else tile_id==water_tile_id
    def get_game_powerup_tiles_dict(self)->dict:
        """Return a dict with powerup keys and walkable tiles list as values."""
        return {"can_debush":[bush_tile_id],"can_swim":[water_tile_id],"can_teleport":[]}
####################
### ACTION SPACE ###
####################
    @lru_cache_func(maxsize=1)
    def get_current_max_action_space(self)->int:
        """Returns the max value of the action space depending on powerups."""
        if self.true_menu:
            return self.allowed_actions_count-1
        max_actions=self.non_powerup_actions_count-1
        if self.get_event_flag("can_debush")>0:
            max_actions=self.action_debush_id
        return max_actions
##################
### RESET-GAME ###
##################
    def game_on_reset(self)->None:
        """Game-specific reset to initial state."""
        for k in ["party_levels","party_hp_ratios"]:
            self.game_state[k][:]=0
        for k in ["money","party_size"]:
            self.game_state[k]=0
##################
### GAME LOGIC ###
##################
    def handle_non_directional_actions(self,action:int)->tuple:
        """Game-specific logic for non-movement buttons."""
        (skip_movement,powerup_started)=(True,False)
        self.game_state["powerup_screen_remove_tile"]=0
        if self.true_menu and action==self.action_menu_id:
            if self.game_state["menu_type"]==0:
                self.game_state["menu_type"]=1
            return (skip_movement,powerup_started)
        if action==self.action_debush_id and self.get_event_flag("can_debush")>0:
            if self.game_state["powerup_walk_tile"]==0:
                self.game_state["powerup_walk_tile"]=bush_tile_id
                self.game_state["powerup_screen_remove_tile"]=bush_tile_id
        elif action==self.action_swim_id and self.get_event_flag("can_swim"):
            if self.game_state["powerup_walk_tile"]!=water_tile_id!=self.game_data["maps"][self.game_state["player_coordinates_data"][0]][self.game_state["player_coordinates_data"][1],self.game_state["player_coordinates_data"][2]] and self.get_faced_tile()==water_tile_id:
                (skip_movement,powerup_started,self.game_state["powerup_walk_tile"])=(False,True,water_tile_id)
        elif action==self.action_teleport_id:
            if self.get_event_flag("can_teleport")>0 and self.secondary_action_value>=0:
                self.powerup_teleport_to(self.secondary_action_value)
        return (skip_movement,powerup_started)
    def game_bypass_powerup_tiles(self,tile:int)->None:
        """Automatic bypass actions of powerups if event prerequisites are met."""
        if tile==bush_tile_id:
            if self.get_event_flag("can_debush")>0:
                self.game_state["powerup_walk_tile"]=bush_tile_id
        elif tile==water_tile_id:
            if self.get_event_flag("can_swim")>0:
                self.game_state["powerup_walk_tile"]=water_tile_id
    def game_powerup_fix_tile(self,tile:int,powerup_started:bool)->int:
        """Fix the tile for powerup-logic purposes."""
        if self.game_state["powerup_walk_tile"]==water_tile_id and powerup_started and not self.is_swimmable_tile_id(tile,True):
            tile=0
        return tile
    def game_powerup_first_time_use_events(self,action:int,tile:int)->None:
        """Set events for first-time powerup usage."""
        if bush_tile_id==self.game_state["powerup_walk_tile"]==tile and self.get_event_flag("can_debush")>0 and self.get_event_flag("first_can_debush")==0:
            self.activate_event_flag("first_can_debush")
        elif water_tile_id==self.game_state["powerup_walk_tile"]==tile and self.get_event_flag("can_swim")>0 and self.get_event_flag("first_can_swim")==0:
            self.activate_event_flag("first_can_swim")
    def game_post_movement_powerup_status(self,tile:int)->None:
        """Finalizes the state of powerup-usage."""
        if self.game_state["powerup_walk_tile"]==bush_tile_id:
            self.game_state["powerup_walk_tile"]=0
            self.game_state["powerup_screen_remove_tile"]=0
        elif water_tile_id==self.game_state["powerup_walk_tile"] and not self.is_swimmable_tile_id(tile,True):
            self.game_state["powerup_walk_tile"]=0
        if self.game_state["powerup_walk_tile"]==0:
            self.game_state["powerup_screen_remove_tile"]=0
    def game_after_step(self,action:int)->None:
        """Runs after the a game step."""
        return
##################################
### AGENTS POWERUP SUGGESTIONS ###
##################################
    def should_use_powerup_or_move_forward(self,can_powerup_name:str):
        """Return 1 if the powerup should be used, 2 to move forward, otherwise 0."""
        if self.get_event_flag(can_powerup_name)>0 and bush_tile_id==self.get_faced_tile():
            return 1 if bush_tile_id!=self.game_state["powerup_walk_tile"] else 2
        return 0
    def should_use_powerup_debush(self)->bool:
        """Return True if the player is facing a bush tile and can remove it."""
        return self.get_event_flag("can_debush")>0 and bush_tile_id==self.get_faced_tile()!=self.game_state["powerup_walk_tile"]
    def should_use_powerup_swim(self)->bool:
        """Return True if the player is facing a water tile and can swim."""
        return self.get_event_flag("can_swim")>0 and water_tile_id==self.get_faced_tile()!=self.game_state["powerup_walk_tile"]
##############################
### HEADLESS BATTLE SYSTEM ###
##############################
    def party_heal(self,min_value:float=1.)->None:
        """Heal the party HP and PP values."""
        if self.game_state["party_size"]<1:
            return
        for i in range(self.game_state["party_size"]):
            self.game_state["party_hp_ratios"][i]=min(1.,max(0.,min_value,self.game_state["party_hp_ratios"][i]))
