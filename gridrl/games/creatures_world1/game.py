#!/usr/bin/env python3

"""Core functions of the game creatures_world1."""

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
try:
    from gridrl.functions_data import (read_json_file,format_child_filename_path,
        is_json_key_comment,convert_hex_str,read_map_metadata_file)
#    from gridrl.functions_numba import nb_in_array_uint8
    from gridrl.core_constants import bush_tile_id
    from gridrl.core_game import lru_cache_func
    from gridrl.core_constants import grass_tile_id,water_tile_id
    from gridrl.abstract_games.exploration_abstract_game import ExplorationAbstractGame
    from gridrl.games.creatures_world1.menu import CreaturesWorld1Menu
    from gridrl.games.creatures_world1.constants import (moves_list,field_moves_list,
        creatures_names,creatures_bst,creatures_evolution_data,
        learn_move_items_dict,learn_move_items_list,key_items_list,
        items_list,items_ids_dict,
    )
except ModuleNotFoundError:
    from functions_data import (read_json_file,format_child_filename_path,
        is_json_key_comment,convert_hex_str,read_map_metadata_file)
#    from functions_numba import nb_in_array_uint8
    from core_constants import bush_tile_id
    from core_game import lru_cache_func
    from core_constants import grass_tile_id,water_tile_id
    from abstract_games.exploration_abstract_game import ExplorationAbstractGame
    from games.creatures_world1.menu import CreaturesWorld1Menu
    from games.creatures_world1.constants import (moves_list,field_moves_list,
        creatures_names,creatures_bst,creatures_evolution_data,
        learn_move_items_dict,learn_move_items_list,key_items_list,
        items_list,items_ids_dict,
    )

__all__=["CreaturesWorld1Game"]

def read_encounters_file(dir_name:str,main_parent:str)->dict:
    """Read the npcs file."""
    data={}
    file_data=read_json_file(format_child_filename_path("wild_encounters.json",dir_name,main_parent))
    if isinstance(file_data,dict) and len(file_data)>0:
        data["wild_encounters"]={convert_hex_str(k):[convert_hex_str(v[0]),np.asarray([[convert_hex_str(s) for s in a[:3]] for a in v[1][:32] if isinstance(a,(list,set,tuple,np.ndarray)) and len(a)>=1],dtype=np.int16)] for k,v in file_data.items()
           if not is_json_key_comment(k) and isinstance(v,(list,set,tuple,np.ndarray)) and len(v)>1 and isinstance(v[0],(int,str)) and isinstance(v[1],(list,set,tuple,np.ndarray)) and len(v)>1
        }
    return data

class CreaturesWorld1Game(ExplorationAbstractGame):
    """The main implementation of the game creatures_world1."""
    has_encoded_assets=True
    def __init__(self,
        config:dict={},
        agent_class:Union[Any,None]=None,
        agent_args:dict={},*args,**kwargs
    )->None:
        """Constructor."""
        game_name="creatures_world1"
###############################
### DEFINING NEW ATTRIBUTES ###
### ALSO DEFINE THEM VIA    ###
### METHODS LISTED BELOW    ###
###############################
        self.start_level=5
        self.invincible=False
        self.trek_map_ids=np.array([0xD9,0xDA,0xDB,0xDC,0xDD,0xDE,0xDF,0xE0,0xE1],dtype=np.uint8)
        self.game_data_creatures_bst=np.array(creatures_bst,dtype=np.uint16,order="C")
        self.game_data_creatures_evolution_data=np.asarray(creatures_evolution_data,dtype=np.uint16,order="C")
#        self.items_list=np.asarray(items_list,dtype=np.uint8,order="C")
        tset=set(field_moves_list)
        self.field_moves_ids=np.array([i for i,k in enumerate(moves_list) if k in tset],dtype=np.uint8,order="C")
        tset=set(learn_move_items_list)
        self.learn_move_items_ids=np.array([i for i,k in enumerate(items_list) if k in tset],dtype=np.uint8,order="C")
        tset=set(key_items_list)
        self.key_items_ids=np.array([i for i,k in enumerate(items_list) if k in tset],dtype=np.uint8,order="C")
        self.puzzle_random_idxs=[12,11]
        self.field_moves_action_ids={}
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
        return CreaturesWorld1Menu
    def game_enforce_config(self,config:dict)->dict:
        """Alter the configurations dict to enforce specific options."""
        return super().game_enforce_config(config)
    def define_game_tile_size(self)->int:
        """Sprite/Tile size in pixels."""
        return 16
    def define_game_screen_tiles_size(self)->tuple:
        """Screen size in tiles unit."""
        return (9,10)
    def define_game_extra_data_functions(self)->list:
        """List of extra game data files functions."""
        return [read_map_metadata_file,read_encounters_file]
    def define_extra_required_game_data_names(self)->list:
        """List of required keys that must exist in the dictionary data."""
        return ["wild_encounters"]
    def define_game_config(self,config:dict)->None:
        """Game-specific configurations initialization."""
        self.start_level=max(5,min(95,int(config.get("start_level",5))))
        self.invincible=bool(config.get("invincible",False))
        if self.sandbox:
            sandbox_start_pos=[0x05,19,15,1]
            self.start_level=40
            self.set_start_positions(start=sandbox_start_pos,checkpoint=sandbox_start_pos)
            self.starting_event=""
            self.starting_collected_flags=["exiting_first_town","start_decision","encounters_tracker"]
            self.starting_collected_flags+=[f"medal{i:d}" for i in range(1,6)]
            self.starting_collected_flags+=[f"powerup_{k[4:]}" for k,_ in self.get_game_powerup_tiles_dict().items()]
    def define_game_config_post_game_state_creation(self,config:dict)->None:
        """Game-specific configurations initialization run after game state declaration."""
        if self.sandbox:
            self.set_first_party_creature()
            self.set_new_party_creature(4,15)
            self.set_item_by_str("powerup_debush")
            self.set_item_by_str("powerup_swim")
            self.add_item_by_id(1,1)
            self.add_item_by_id(2,2)
            self.add_item_by_id(3,4)
            self.add_item_by_id(1,10)
            self.drop_item_by_id(3,1)
            self.add_item_by_id(4,4)
            self.drop_item_by_id(2,0xFF)
    def define_internal_data(self)->None:
        """Game-specific attribute declarations."""
        self.trek_map_ids=np.array([0xD9,0xDA,0xDB,0xDC,0xDD,0xDE,0xDF,0xE0,0xE1],dtype=np.uint8)
        self.game_data_creatures_bst=np.array(creatures_bst,dtype=np.uint16,order="C")
        self.game_data_creatures_evolution_data=np.asarray(creatures_evolution_data,dtype=np.uint16,order="C")
        self.puzzle_random_idxs=[12,11]
        super().define_internal_data()
    def define_actions_ids(self)->int:
        """Custom game actions id declaration."""
        ret=super().define_actions_ids()
        self.field_moves_action_ids={
            "debush":self.action_debush_id,
            "swim":self.action_swim_id,
        }
        return ret
    def define_extra_game_state(self)->dict:
        """Dict of extra values preserved in the game_state dictionary."""
        state=super().define_extra_game_state()
        state.update({"money":3000,"party_size":0,"bag_size":0,
            "encounter_steps":0,"trek_timeout":0,
            "bag":np.zeros((20,2),np.uint8,order="C"),
            "tracker_flags":np.zeros((len(creatures_names),),np.uint8,order="C"),
            "party_index":np.zeros((6,),np.int16,order="C"),
            "party_moves":np.zeros((6,4),np.int16,order="C"),
            "party_levels":np.zeros((6,),np.uint8,order="C"),
            "party_exps":np.zeros((6,),np.uint32,order="C"),
            "party_bsts":np.zeros((6,),np.uint16,order="C"),
            "party_avg_stats":np.zeros((6,),np.uint16,order="C"),
            "party_hp_ratios":np.zeros((6,),np.float32,order="C"),
            "party_pp_ratios":np.zeros((6,),np.float32,order="C"),
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
        }
#####################
### MAP FUNCTIONS ###
#####################
    @lru_cache_func(maxsize=2048)
    def cached_is_trek_map(self,map_id:int)->bool:
        """Is the map matching any trek zone."""
        return map_id in self.trek_map_ids
#        return nb_in_array_uint8(map_id,self.trek_map_ids)
    @lru_cache_func(maxsize=256)
    def is_exp_map_tile(self,tile_id:int)->bool:
        """Return True if the tile can generate encounters."""
        return tile_id in [grass_tile_id,water_tile_id]
    @lru_cache_func(maxsize=2048)
    def is_exp_map_regardless_tile(self,map_id:int)->bool:
        """Return True if the map can generate encounters regardless the tile."""
        return map_id in [0x3B,0x3C,0x3D,0xC5,0x52,0xE8,0xA5,0xD6,0xD7,0xD8,0x8E,0x8F,0x90,0x91,0x92,0x93,0x94,0x53,0xC0,0x9F,0xA0,0xA1,0xA2,0x6C,0xC2,0xC6,0xE2,0xE3,0xE4]
######################
### EVENT-HANDLING ###
######################
    def define_game_critical_event_names(self)->None:
        """Define first and last events of the game."""
        self.first_event_name="exiting_first_town"
        self.trigger_done_event_name="megaphone"
    def game_on_event_custom_load(self,starting_event:str,used_collected_flags:set,used_level:int)->None:
        """Game-specific custom state load fixes."""
        if starting_event not in ["",self.first_event_name,"start_decision"]:
            self.set_first_party_creature(self.start_level if "start_decision" not in used_collected_flags else max(used_level,self.start_level))
    def game_on_event_flag_change(self,event_name:str,activated:bool)->bool:
        """Used to for validating event states. Return True to clear again the event cache."""
        if event_name=="start_decision" and activated and  self.game_state["party_size"]<1:
            self.set_first_party_creature()
            return True
        return False
##################
### DEBUG TEXT ###
##################
    def get_party_text(self)->str:
        """Party summary text."""
        return f"Level: {self.game_state['party_levels'][0]:3d}\tHP: {self.game_state['party_hp_ratios'][0]:.0%}\tPP: {self.game_state['party_pp_ratios'][0]:.0%}"
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
##################
### RESET-GAME ###
##################
    def reset_party(self)->None:
        """Initialize the party data."""
        for k in ["party_index","party_moves","party_levels","party_exps","party_bsts","party_avg_stats","party_hp_ratios","party_pp_ratios"]:
            self.game_state[k][:]=0
        self.game_state["party_size"]=0
    def reset_bag(self)->None:
        """Initialize the bag data."""
        self.game_state["bag"][:]=0
        self.game_state["bag_size"]=0
    def game_on_reset(self)->None:
        """Game-specific reset to initial state."""
        self.game_state["encounter_steps"]=0
        self.game_state["money"]=3000
        self.reset_party()
        self.reset_bag()
###########################
### STRUCTURE FUNCTIONS ###
###########################
    def switch_party_creatures(self,pos1:int,pos2:int)->None:
        """Switch order of creatures in the party."""
        for k in ["party_index","party_moves","party_levels","party_exps","party_bsts","party_avg_stats","party_hp_ratios","party_pp_ratios"]:
            t=self.game_state[k][pos1].copy()
            self.game_state[k][pos1]=self.game_state[k][pos2]
            self.game_state[k][pos2]=t
    def learnable_move_item_to_move_id(self,idx:int)->int:
        """Return the id of the move contained in the item."""
        return learn_move_items_dict.get(items_list[idx],0)
    def learnable_move_item_to_move_name(self,idx:int)->str:
        """Return the name of the move contained in the item."""
        return moves_list[self.learnable_move_item_to_move_id(idx)]
    def learn_move_to_creature(self,party_pos:int,move_id:int,move_pos:int)->bool:
        """Override move slot on the creature"""
        self.game_state["party_moves"][party_pos,move_pos]=move_id
        return True
    def get_item_id_from_str(self,name:str)->int:
        """String to item id dict lookup."""
        return items_ids_dict.get(name,0)
    def get_item_id_by_pos(self,pos:int)->int:
        """Return the id of the item given its bag position."""
        return self.game_state["bag"][pos,0]
    def get_item_bag_quantity_by_pos(self,pos:int)->int:
        """Return the quantity of the item given its bag position."""
        return self.game_state["bag"][pos,1]
    def get_item_pos_in_bag(self,idx:int)->int:
        """Return the bag index of the item in bag or -1 when not found."""
        for i in range(self.game_state["bag_size"]):
            if self.game_state["bag"][i,0]==idx:
                return i
        return -1
    def set_item_by_pos(self,idx:int,pos:int,amount:int=1)->bool:
        """Set item id and amount given a bag position."""
        if amount<1:
            return self.drop_item_by_pos(pos,0xFF)
        if pos<0:
            pos=self.game_state["bag_size"]
            if self.game_state["bag_size"]>=self.game_state["bag"].shape[0]:
                return False
            self.game_state["bag"][pos,0]=idx
            self.game_state["bag"][pos,1]=amount
            self.game_state["bag_size"]+=1
        else:
            self.game_state["bag"][pos,1]=amount
        return True
    def set_item_by_id(self,idx:int,amount:int=1)->bool:
        """Set item id and amount."""
        return self.set_item_by_pos(idx,self.get_item_pos_in_bag(idx),amount)
    def set_item_by_str(self,name:str,amount:int=1)->bool:
        """Set item from string."""
        idx=self.get_item_id_from_str(name)
        return self.set_item_by_pos(idx,self.get_item_pos_in_bag(idx),amount) if idx>0 else False
    def add_item_by_pos(self,idx:int,pos:int,amount:int=1)->bool:
        """Increase item in bag by position."""
        if pos<0:
            pos=self.game_state["bag_size"]
            if self.game_state["bag_size"]>=self.game_state["bag"].shape[0]:
                return False
            prev_val=self.game_state["bag"][pos,1]
            self.game_state["bag"][pos,0]=idx
            self.game_state["bag"][pos,1]+=amount
            if prev_val>self.game_state["bag"][pos,1]:
                self.game_state["bag"][pos,1]=0xFF
            self.game_state["bag_size"]+=1
        else:
            self.game_state["bag"][pos,1]+=amount
        return True
    def add_item_by_id(self,idx:int,amount:int=1)->bool:
        """Increase item in bag by id."""
        return self.add_item_by_pos(idx,self.get_item_pos_in_bag(idx),amount)
    def add_item_by_str(self,name:str,amount:int=1)->bool:
        """Increase item from string."""
        idx=self.get_item_id_from_str(name)
        return self.add_item_by_pos(idx,self.get_item_pos_in_bag(idx),amount) if idx>0 else False
    def drop_item_by_pos(self,pos:int,amount:int=0xFF)->bool:
        """Decrease item in bag by position."""
        if pos<0:
            return False
        prev_val=self.game_state["bag"][pos,1]
        self.game_state["bag"][pos,1]-=amount
        if prev_val<self.game_state["bag"][pos,1]:
            self.game_state["bag"][pos,1]=0
        if self.game_state["bag"][pos,1]<1:
            self.game_state["bag"][pos,0]=0
            self.game_state["bag"][pos:self.game_state["bag_size"]]=np.roll(self.game_state["bag"][pos:self.game_state["bag_size"]],-1,axis=0)
            self.game_state["bag_size"]-=1
        return True
    def drop_item_by_id(self,idx:int,amount:int=0xFF)->bool:
        """Decrease item in bag by id."""
        return self.drop_item_by_pos(self.get_item_pos_in_bag(idx),amount)
    def drop_item_by_str(self,name:str,amount:int=0xFF)->bool:
        """Increase item in bag by position."""
        idx=self.get_item_id_from_str(name)
        return self.drop_item_by_pos(self.get_item_pos_in_bag(idx),amount) if idx>0 else False
    def use_item(self,item_pos:int)->None:
        """Decrease item from string."""
        return
    def use_item_on_creature(self,item_pos:int,party_pos:int)->None:
        """Apply the item on the creature."""
        return
    def use_item_on_move(self,item_pos:int,party_pos:int,move_pos:int)->None:
        """Apply the item on the move of a creature."""
        return
    def use_field_move(self,move_id)->bool:
        """Binds usage of the field move withing the environment."""
        if move_id in self.field_moves_ids:
            action_id=self.field_moves_action_ids.get(moves_list[move_id],0)
            if action_id>0:
                self.add_forced_action(action_id)
                return True
        return False
###############################
### STRUCTURES CONDITIONALS ###
###############################
    def get_creature_field_moves(self,pos)->list:
        """Gets ids of field moves on a given creature."""
        return [k for k in self.env.game_state["party_moves"][pos] if k in self.field_moves_ids] if pos<self.env.game_state["party_size"] else []
    def can_use_field_move(self,move_id)->None:
        """Is allowed to use the field move."""
        if move_id in self.field_moves_ids:
            return self.get_event_flag(f"can_{moves_list[move_id]}")>0
        return False
    def can_toss_item_by_pos(self,pos:int)->bool:
        """Return if the item can be tossed safely."""
        return not self.is_key_item(self.get_item_id_by_pos(pos))
    def can_use_item(self,item_pos:int)->bool:
        """Check if the item can be used."""
        return True
    def creature_has_free_move_slot(self,party_pos:int)->bool:
        """Check if the creature has a free move slot."""
        return False
    def can_use_item_on_creature(self,item_pos:int,party_pos:int)->bool:
        """Return if the item can be used on the creature."""
        return True
    def can_use_move_on_creature(self,item_pos:int,party_pos:int)->bool:
        """@."""
        return True
    def is_duplicate_move_on_creature(self,item_pos:int,party_pos:int)->bool:
        """@."""
        return False
    def item_used_on_creature(self,item_pos:int,party_pos:int)->bool:
        """@."""
        return True
    def is_pp_cure(self,idx:int)->bool:
        """@."""
        return False
    def is_overworld_item(self,idx:int)->bool:
        """@."""
        return False
    def is_key_item(self,idx:int)->bool:
        """Return if the item is a key item."""
        for k in self.key_items_ids:
            if idx==k:
                return True
        return False
    def is_teachable_move(self,idx:int)->bool:
        """Return if the item is a teachable move."""
        for k in self.learn_move_items_ids:
            if idx==k:
                return True
        return False
    def is_repel(self,item_pos:int)->bool:
        """@."""
        return False
    def get_creature_moves_count(self,party_pos:int)->int:
        """Return the amount of moves of a creature."""
        return 4
    def move_forgettable_by_creature(self,party_pos:int,move_id:int,move_pos:int)->bool:
        """Check if a powerup move is going to be overridden by a new move."""
        return self.env.game_state["party_moves"][party_pos,move_pos] not in self.field_moves_ids
#########################
### RANDOM ENCOUNTERS ###
#########################
    def pseudo_random_encounter_chance(self,off:int=0)->bool:
        """Faster pseudo-random chance. Return True to get an encounter."""
        return (self.step_count//256+self.game_state["player_coordinates_data"][0]+self.game_state["encounter_steps"]+self.game_state["pseudo_seed"]+off)%10==9
    def game_handle_random_encounter_spawn(self,map_id:int,tile:int)->int:
        """Handles random encounters. Return 0 without encounters, 1 on win, -1 on loss"""
        if self.game_state["party_size"]>0 and map_id in self.game_data["wild_encounters"] and (
            self.is_exp_map_tile(tile) or self.is_exp_map_regardless_tile(map_id)
        ):
            self.game_state["encounter_steps"]+=1
            if self.pseudo_random_encounter_chance():
                return 1 if self.headless_natural_battle(self.get_random_encounter_level(map_id)) else -1
        return 0
########################
### GAME STATE-CHECK ###
########################
    def check_creatures_owned(self,count:int)->bool:
        """Checks if the amount of creatures ever owned is not lower than the input."""
#        return np.ndarray.sum(self.game_state["slots_owned"])>=count
        return True
##############################
### HEADLESS BATTLE SYSTEM ###
##############################
    def party_heal(self,min_value:float=1.)->None:
        """Heal the party HP and PP values."""
        if self.game_state["party_size"]<1:
            return
        for i in range(self.game_state["party_size"]):
            self.game_state["party_hp_ratios"][i]=min(1.,max(0.,min_value,self.game_state["party_hp_ratios"][i]))
            self.game_state["party_pp_ratios"][i]=min(1.,max(0.,min_value,self.game_state["party_pp_ratios"][i]))
    @lru_cache_func(maxsize=128)
    def exp_formula_basic(self,level:int)->int:
        """Formula for level-up exp."""
        return max(9,int(6/5*level**3)-15*level**2+100*level+140)
    @lru_cache_func(maxsize=512)
    def get_headless_damage_formula(self,attacker_level:int,attacker_stat:int,defender_level:int,stat_defender:int,critical:bool=False)->float:
        """Simple damage formula."""
        if attacker_level<4:
            move_power=30
        elif attacker_level<8:
            move_power=40
        elif attacker_level<13:
            move_power=50
        elif attacker_level<20:
            move_power=60
        elif attacker_level<32:
            move_power=70
        elif attacker_level<45:
            move_power=80
        else:
            move_power=95
        hp=float(stat_defender)*1.5
        return max(2,(((2+(2*float(attacker_level)*(1.5 if critical else 2))/5.)*float(move_power)*(float(attacker_stat)/float(stat_defender)))/50.+2))/hp
    @lru_cache_func(maxsize=512)
    def get_encounter_exp(self,level:int,base_exp:int=64,natural:bool=True,traded:bool=False)->int:
        """Formula for encounter exp."""
        return int((1 if natural else 1.5)*(1.5 if traded else 1)*base_exp*level/7.)
    def set_new_party_creature(self,creature_id:int,level:int=2)->None:
        """Assigns a new creature to the party."""
        if self.game_state["party_size"]>=len(self.game_state["party_hp_ratios"]):
            return
        idx=self.game_state["party_size"]
        self.env.game_state["tracker_flags"][creature_id]=2
        self.set_creature_level(idx,max(1,level))
        self.game_state["party_index"][idx]=creature_id
        self.game_state["party_bsts"][idx]=self.game_data_creatures_bst[idx]
        self.game_state["party_avg_stats"][idx]=self.calculate_creature_stats(self.game_state["party_levels"][idx],self.game_state["party_bsts"][idx])
        self.game_state["party_hp_ratios"][idx]=1.
        self.game_state["party_pp_ratios"][idx]=1.
        self.game_state["party_size"]+=1
        self.game_state["player_level"]+=self.game_state["party_levels"][idx]
        for i in range(4):
            self.game_state["party_moves"][idx,i]=i+2
        for _ in range(2):
            if not self.check_stats_evolution(idx):
                break
    def set_first_party_creature(self,level:int=0)->None:
        """Assigns the first creature to the party."""
        if self.game_state["party_size"]<1:
            self.set_new_party_creature(1,max(5,self.start_level if level==0 else level))
    def swap_party_creatures(self,pos1:int,pos2:int)->None:
        """Swap party position."""
        return
    @lru_cache_func(maxsize=512)
    def calculate_creature_stats(self,level:int,bst:int,dv_mult:float=0.,medals:int=0)->int:
        """Formula for internal creature stat."""
        return int((1+float(medals)/64.)*(5+float(bst)/5.*float(level)/50.+dv_mult*93*float(level)/100.))
    def check_stats_evolution(self,idx:int)->bool:
        """Check if the creature can transform into a new form."""
        evo_data=self.game_data_creatures_evolution_data[self.game_state["party_index"][idx]]
        if self.game_state["party_levels"][idx]>=evo_data[0]>0 and evo_data[1]==1:
            prev_stat=self.game_state["party_avg_stats"][idx]
            self.game_state["party_index"][idx]=evo_data[2]
            self.game_state["party_bsts"][idx]=self.game_data_creatures_bst[evo_data[2]]
            self.game_state["party_avg_stats"][idx]=self.calculate_creature_stats(self.game_state["party_levels"][idx],self.game_state["party_bsts"][idx])
            self.game_state["party_hp_ratios"][idx]=max(1e-4,min(1.,(self.game_state["party_hp_ratios"][idx]*prev_stat+self.game_state["party_avg_stats"][idx]-prev_stat)/self.game_state["party_avg_stats"][idx]))
            return True
        return False
    def set_creature_level(self,idx:int,level,set_exp:bool=True,set_hp:bool=True)->None:
        """Set the creature level and update stats."""
        prev_stat=self.game_state["party_avg_stats"][idx]
        self.game_state["party_levels"][idx]=max(0,min(100,int(level)))
        if idx==0:
            self.game_state["player_level"]=self.game_state["party_levels"][idx]
        if set_exp:
            self.game_state["party_exps"][idx]=self.exp_formula_basic(self.game_state["party_levels"][idx])
        self.game_state["party_avg_stats"][idx]=self.calculate_creature_stats(self.game_state["party_levels"][idx],self.game_state["party_bsts"][idx])
        if set_hp:
            self.game_state["party_hp_ratios"][idx]=max(1e-4,min(1.,(self.game_state["party_hp_ratios"][idx]*prev_stat+self.game_state["party_avg_stats"][idx]-prev_stat)/self.game_state["party_avg_stats"][idx]))
    def level_up_creature(self,idx:int,set_exp:bool=True,set_hp:bool=True,battle_ended:bool=True)->None:
        """Raise creature to the next level."""
        check_evo=self.game_state["party_levels"][idx]<100 and battle_ended
        self.set_creature_level(idx,self.game_state["party_levels"][idx]+1,set_exp,set_hp)
        if check_evo:
            self.check_stats_evolution(idx)
    def gain_exp(self,idx:int,exp,battle_ended:bool=True)->None:
        """Increase creature exp and handle any level-up."""
        if self.game_state["party_levels"][idx]<1 or self.game_state["party_levels"][idx]>=100:
            return
        self.game_state["party_exps"][idx]+=exp
        for _ in range(100):
            if self.game_state["party_exps"][idx]>=self.exp_formula_basic(self.game_state["party_levels"][idx]+1):
                self.level_up_creature(idx,False,True,battle_ended)
                if self.game_state["party_levels"][idx]>=100:
                    self.game_state["party_exps"][idx]=self.exp_formula_basic(100)
                    break
            else:
                break
    def get_random_encounter_level(self,map_id:int)->int:
        """Return the level of the encounter."""
        try:
            return self.game_data["wild_encounters"].get(map_id,[0,[[0,2,1]]])[1][0][0]
        except IndexError:
            return 2
    def headless_battle_npc(self,level:int,num:int=1,moves_penalty:float=1.,only_once:bool=True)->bool:
        """Battles an NPC."""
        return self.headless_battle_npc_internal(self.last_npc_name,level,num,moves_penalty,only_once)
    def headless_battle_npc_internal(self,npc_name:str,level:int,num:int=1,moves_penalty:float=1.,only_once:bool=True)->bool:
        """Internal NPC integration with a battle result."""
        if only_once and self.get_npc_configs(npc_name)["won"]>0:
            return True
        ret=self.headless_battle_internal(level,num,moves_penalty,natural=False)
        if self.using_npcs and ret:
            self.modify_npc_configs(npc_name,{"won":1})
        return ret
    def headless_natural_battle(self,level:int,num:int=1,moves_penalty:float=1.)->bool:
        """Battles not triggered by NPC."""
        return self.headless_battle_internal(level,num,moves_penalty,natural=True)
    def headless_battle_internal(self,level:int,num:int=1,moves_penalty:float=1.,natural:bool=False)->bool:
        """Main battle logic."""
        lost=False
        for i in range(1 if natural or num<1 else min(6,int(num))):
            levels=[self.game_state["party_levels"][0],level]
            stats=[self.game_state["party_avg_stats"][0],self.calculate_creature_stats(level,300)]
            hps=[self.game_state["party_hp_ratios"][0],1.]
            pps=[self.game_state["party_pp_ratios"][0],1.]
            damage_mults=[1.,0.9*max(0.25,min(1.,moves_penalty))]
            opponent_defeated=False
            for _ in range(50):
                for idx in range(2):
###                 print(self.get_headless_damage_formula(levels[idx],stats[idx],levels[idx^1],stats[idx^1],critical=False))
                    damage=damage_mults[idx]*self.get_headless_damage_formula(levels[idx],stats[idx],levels[idx^1],stats[idx^1],critical=False)
                    if pps[idx]>0.:
                        hps[idx^1]-=damage
                        pps[idx]-=0.01+levels[idx]/15000.
                    else:
                        hps[idx]-=0.1
                        hps[idx^1]-=0.25*damage
                    if hps[0]<=0 or hps[1]<=0:
                        opponent_defeated=True
                        break
                if opponent_defeated:
                    break
###         print(hps,stats,j)
            self.game_state["party_hp_ratios"][0]=max(-1e-6,hps[0]-min(0.1,max(5,20+levels[1]-levels[0])/2500.))
            self.game_state["party_pp_ratios"][0]=max(0.,pps[0])
            if hps[0]<=0:
                lost=True
                if not self.invincible:
                    break
            self.gain_exp(0,self.get_encounter_exp(level,natural=natural),False)
        heal_skip_loss=self.game_state["player_coordinates_data"][0]==0x28
        if heal_skip_loss:
            self.party_heal()
        elif lost:
            self.game_state["battle_type"]=0
            self.close_menu()
            self.party_heal(0.2 if self.invincible else 1.)
            if not self.invincible:
                self.game_state["loss_count"]+=1
                self.game_state["money"]=self.game_state["money"]//2
                self.teleport_to_checkpoint()
        if not lost and not natural:
            self.game_state["money"]=max(999999,self.game_state["money"]+level*40)
        if self.invincible or (not lost and self.using_npcs):
            for i in range(self.game_state["party_size"]):
                self.check_stats_evolution(i)
        return not lost
