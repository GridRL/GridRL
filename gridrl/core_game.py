#!/usr/bin/env python3

"""GameCore core implementation."""

from typing import Union,Any
from collections import deque,OrderedDict
from copy import deepcopy
import warnings
import struct
import sys
import time
from io import BytesIO
import numpy as np
sys.dont_write_bytecode=True
warnings.filterwarnings("ignore",category=DeprecationWarning)

if __package__ is None or len(__package__)==0:
    from configs_speedup_dependencies import lru_cache_func
    from functions_data import get_function_from_self_or_scope,pickle_load,pickle_save,filter_dict_key_depending_on_npc
    from functions_images import tile_split,fill_pad_image,rgb_to_grayscale,map_matrix_to_image,generate_characters_tiles,generate_gif_from_numpy,show_image
    from functions_port_dependencies import rgb_to_hsv,hsv_to_rgb,downscale_local_mean,to_categorical
#    from functions_numba import initialize_numba_functions,nb_is_point_inside_area#,nb_in_array_uint8,nb_assign_uint8_2d,nb_assign_int16,nb_assign_int16_2d,nb_sum_int16
    from game_module_selector import GameModuleSelector
    from core_constants import (shrinked_characters_list,
        no_map_tile_id,unwalkable_tile_id,walk_tile_id,
        warp_tile_id,
        ledge_down_tile_id,ledge_left_tile_id,ledge_right_tile_id,ledge_up_tile_id,
        water_tile_id,bush_tile_id,
        indoor_puzzle_platform_reset_tile_id,indoor_puzzle_platform_down_tile_id,
        indoor_puzzle_platform_left_tile_id,indoor_puzzle_platform_right_tile_id,
        indoor_puzzle_platform_up_tile_id,
        water_current_down_tile_id,water_current_left_tile_id,
        water_current_right_tile_id,water_current_up_tile_id,
        active_script_tile_id,old_script_tile_id,active_reward_tile_id,old_reward_tile_id,
        player_down_tile_id,npc_down_tile_id,npc_left_tile_id,npc_right_tile_id,npc_up_tile_id,
        generic_menu_tile_id,
        generic_cursor_tile_id,
        count_tiles_ids,
        walkable_tiles_ids,tile_id_colors_list,
    )
else:
    from gridrl.configs_speedup_dependencies import lru_cache_func
    from gridrl.functions_data import get_function_from_self_or_scope,pickle_load,filter_dict_key_depending_on_npc
    from gridrl.functions_images import tile_split,fill_pad_image,rgb_to_grayscale,map_matrix_to_image,generate_characters_tiles,generate_gif_from_numpy,show_image
    from gridrl.functions_port_dependencies import rgb_to_hsv,hsv_to_rgb,downscale_local_mean,to_categorical
#    from gridrl.functions_numba import initialize_numba_functions,nb_is_point_inside_area#,nb_in_array_uint8,nb_assign_uint8_2d,nb_assign_int16,nb_sum_int16
    from gridrl.game_module_selector import GameModuleSelector
    from gridrl.core_constants import (shrinked_characters_list,
        no_map_tile_id,unwalkable_tile_id,walk_tile_id,
        warp_tile_id,
        ledge_down_tile_id,ledge_left_tile_id,ledge_right_tile_id,ledge_up_tile_id,
        water_tile_id,bush_tile_id,
        indoor_puzzle_platform_reset_tile_id,indoor_puzzle_platform_down_tile_id,
        indoor_puzzle_platform_left_tile_id,indoor_puzzle_platform_right_tile_id,
        indoor_puzzle_platform_up_tile_id,
        water_current_down_tile_id,water_current_left_tile_id,
        water_current_right_tile_id,water_current_up_tile_id,
        active_script_tile_id,old_script_tile_id,active_reward_tile_id,old_reward_tile_id,
        player_down_tile_id,npc_down_tile_id,npc_left_tile_id,npc_right_tile_id,npc_up_tile_id,
        generic_menu_tile_id,
        generic_cursor_tile_id,
        count_tiles_ids,
        walkable_tiles_ids,tile_id_colors_list,
    )

__all__=["GameAbstractCore","GameCore","lru_cache_func"]

class GameAbstractCore:
    """The abstract GameCore class used for Environment subclassing."""
    has_encoded_assets=False
    def __init__(self):
        self.movement_max_actions=4
        self.all_actions_count=4
        self.screen_observation_type=0
        self.auto_screen_obs=True
        self.log_screen=False
        self.game_completed=False
#        initialize_numba_functions()
#############################
### GAME ABSTRACT METHODS ###
#############################
    def env_is_done(self)->bool:
        """Extra conditions to the game execution."""
        return False
    def env_reward(self,action:int=-1)->float:
        """Total reward of the action at the current step."""
        return 5.*self.get_event_flags_sum()+25.*self.game_completed
    def get_nonscreen_observations(self)->dict[np.ndarray]:
        """Main method to declare the step non-screen observations."""
        return {}
    def get_nonscreen_observation_space(self)->dict:
        """Declarations of the get_nonscreen_observations space types."""
        return {}
    def update_reward(self,action:int=-1)->float:
        """Delta reward of the action at the current step."""
        return 0.
##################################
### GAME CORE ABSTRACT METHODS ###
##################################
    def predict_action(self)->int:
        """Predict an action if there is an agent linked or pick one randomly otherwise."""
        return 0
    def get_screen_observation(self)->dict[np.ndarray]:
        """Return the screen ndarray according to the screen_observation_type config."""
        return np.zeros((1,1),dtype=np.uint8,order="C")
    def get_observations(self)->dict[np.ndarray]:
        """Return the observation dictionary."""
        obs=self.get_nonscreen_observations()
        if self.auto_screen_obs:
            obs["screen"]=self.get_screen_observation()
        return obs
    def get_event_flags_sum(self)->int:
        """Return the sum of all event flags."""
        return 0
    def reset_game(self,seed:Union[int,None]=None)->None:
        """Reset the game to the initial state."""
        return
    def run_action_on_emulator(self,action:int=-1)->int:
        """Process one game frame."""
        return 0
    def step_game(self,action:int=-1)->int:
        """Step the game one frame. Safer function wrapper."""
        return 0
    def is_done(self)->bool:
        """Return True to stop the game execution."""
        return False
    def close(self)->None:
        """Cleanup code for freeing environment resources."""
        return

class GameCore(GameAbstractCore):
    """The main GameCore class handling the game logic."""
    def __init__(self,
        game_name:str,
        config:Union[dict,None]=None,
        agent_class:Union[Any,None]=None,
        agent_args:Union[dict,None]=None,
        *args,**kwargs
    )->None:
        """Constructor."""
        GameAbstractCore.__init__(self)
        if config is None:
            config={}
        if agent_args is None:
            agent_args={}
        self.env=self
        self.env_config=self.game_enforce_config(deepcopy(config))
        ignore_custom=bool(config.get("ignore_custom",False))
        self.game_selector=GameModuleSelector(game_name=game_name,ignore_custom=ignore_custom)
        self.scripting_core_scope=self.game_selector.fallback_get_scripting_module()
        self.agent=None
        self.menu=None
        self.seed=0
        self.step_count=0
        self.game_completed=False
        self.step_recursion_depth=0
        self.total_reward=0.
        self.act_freq=1
        self.called_step_func_names=set()
        self.default_monocromatic_screen_scale=3
        self.from_gui=bool(config.get("from_gui",False))
        self.benchmarking=bool(config.get("benchmarking",False))
        self.skip_validation=bool(config.get("skip_validation",True))
        self.auto_screen_obs=bool(config.get("auto_screen_obs",True))
        self.rewind_steps=max(0,int(config.get("rewind_steps",0)))
        grayscale_screen=bool(config.get("grayscale_screen",False))
        grayscale_single_channel=bool(config.get("grayscale_single_channel",True))
        self.use_gfx_image=bool(config.get("gfx_image",False))
        self.hsv_image=bool(config.get("hsv_image",False)) and not grayscale_screen
        self.screen_downscale=max(1,min(8,int(config.get("screen_downscale",1))))
        self.tile_size=self.define_game_tile_size()
        self.scaled_sprites_size=self.tile_size//self.screen_downscale
        self.action_nop=bool(config.get("action_nop",False))
        self.starting_event=config.get("starting_event","")
        self.starting_collected_flags=config.get("starting_collected_flags",[])
        self.strip_ledges=bool(config.get("strip_ledges",False))
        self.movement_max_actions=max(4,min(2,int(config.get("movement_max_actions",4))))
        self.bypass_powerup_actions=bool(config.get("bypass_powerup_actions",False))
        self.button_interaction=bool(config.get("button_reward",False))
        self.using_npcs=bool(config.get("using_npcs",False))
        self.true_menu=bool(config.get("true_menu",False))
        self.action_complexity=max(-2,min(4,int(config.get("action_complexity",-1))))
        self.screen_observation_type=max(-1,min(4,int(config.get("screen_observation_type",-1))))
        menu_class=self.get_menu_class()
        self.rewind_states_deque=deque(maxlen=1024)
        self.last_action=0
        self.text_queue=[]
        self.text_changed_during_step=True
        self.menu_changed_during_step=True
        self.menu_current_depth=-1
        self.menu_content_by_depth=[[] for i in range(8)]
        self.menu_bg_by_depth=np.zeros((8,5),dtype=np.int16,order="C")
        self.menu_cursor_data=np.zeros((8,7,),dtype=np.int16,order="C")
        self.last_gfx_menu_overlap=np.zeros((1,),dtype=np.uint8,order="C")
        self.shrinked_characters_list=deepcopy(shrinked_characters_list)
        self.convertion_dict={"A":"A","B":"T","C":"H","D":"U","E":"E","F":"N","G":"I",
            "H":"H","I":"I","J":"E","K":"A","L":"R","M":"H","N":"N",
            "O":"O","P":"T","Q":"U","R":"R","S":"S","T":"T",
            "U":"U","V":"A","W":"S","X":"U","Y":"O","Z":"U",
            " ":"D","\t":"D","\r":"D","\n":"D","-":"L",
            "0":"U","1":"E","2":"A","3":"T","4":"O",
            "5":"I","6":"N","7":"S","8":"H","9":"R"}
        self.encoded_characters_lookup=np.full((256,),len(self.shrinked_characters_list)-1,dtype=np.uint8,order="C")
        for k,v in self.convertion_dict.items():
            self.encoded_characters_lookup[ord(k)]=self.shrinked_characters_list.index(v)
            self.encoded_characters_lookup[ord(k.lower())]=self.shrinked_characters_list.index(v)
        for _ in range(2):
            if self.action_complexity==-1:
                if self.true_menu:
                    self.action_complexity=2 if menu_class is None else 3
                elif self.using_npcs:
                    self.action_complexity=2
                elif self.button_interaction:
                    self.action_complexity=1
                elif self.bypass_powerup_actions:
                    self.action_complexity=0
                else:
                    self.action_complexity=1
            if self.screen_observation_type==-1:
                self.screen_observation_type=4 if self.use_gfx_image or self.hsv_image else 2
            if self.action_complexity>2 or self.screen_observation_type in (3,4):
                self.use_gfx_image=self.screen_observation_type==4
            if self.action_complexity==0:
                (self.bypass_powerup_actions,self.button_interaction,self.using_npcs,self.true_menu)=(True,False,False,False)
            elif self.action_complexity==1:
                (self.bypass_powerup_actions,self.button_interaction,self.using_npcs,self.true_menu)=(False,True,False,False)
            elif self.action_complexity==2:
                (self.bypass_powerup_actions,self.button_interaction,self.using_npcs,self.true_menu)=(False,True,True,False)
            elif self.action_complexity>2:
                (self.movement_max_actions,self.bypass_powerup_actions,self.button_interaction,self.using_npcs,self.true_menu)=(4,False,True,True,True)
            if self.use_gfx_image and not self.using_npcs:
                self.screen_observation_type=max(0,min(3,self.screen_observation_type))
                self.action_complexity=max(0,min(1,self.action_complexity))
                self.gfx_loaded=False
                self.use_gfx_image=False
            else:
                break
        if self.true_menu:
            menu_class=self.get_menu_class()
            if menu_class is None:
                self.true_menu=False
            else:
                self.menu=menu_class(self)
        self.log_screen=bool(config.get("log_screen",False))
        self.gif_speedup=int(config.get("gif_speedup",-1))
        self.max_steps=int(max(-1,config.get("max_steps",2**31)))
        self.no_assets=bool(config.get("no_assets",False))
        self.totally_headless=self.screen_observation_type==0 and self.auto_screen_obs and not self.log_screen
        self.action_directions=np.array([0,2,3,1],np.uint8)
        self.inverted_directions=np.array([1,0,3,2,4],np.uint8)
        self.directions_offsets=np.asarray([[1,0],[-1,0],[0,-1],[0,1],[0,0]],dtype=np.int16)
        self.movement_offsets=np.asarray([[1,0],[0,-1],[0,1],[-1,0],[0,0]],dtype=np.int16)
        screen_view_mult=int(max(1,config.get("screen_view_mult",1)))
        game_screen_sizes=np.array(self.define_game_screen_tiles_size()[:2],np.int16)*screen_view_mult
        for i,v in enumerate(["screen_y","screen_x"]):
            game_screen_sizes[i]=int(config.get(v,game_screen_sizes[i]))
        self.centered_screen_size=game_screen_sizes.clip(1,0x100)
        self.player_screen_position=(self.centered_screen_size-1)//2
        self.player_screen_bounds=np.vstack([-self.player_screen_position,self.centered_screen_size-self.player_screen_position],dtype=np.int16).transpose(1,0).flatten()
        self.start_map_position=np.zeros((4,),dtype=np.int16,order="C")
        self.first_checkpoint_place=np.zeros((4,),dtype=np.int16,order="C")
        self.gif_frames=[]
        self.game_data=self.game_selector.fallback_read_all_data_files()
        if not self.benchmarking:
            required_game_data_names=set(["channels","sizes","legacy_global_starting_coords","warps","maps","teleport_data","events","scripts","npcs","powerups"])
            required_game_data_names=required_game_data_names.union(set(self.define_extra_required_game_data_names()))
            assert len(self.game_data)>=len(required_game_data_names) and len(required_game_data_names-set(self.game_data.keys()))==0
        self.padding_fixed=False
        self.global_map_shapes=np.ones((3,),dtype=np.int16,order="C")
        self.global_map_padding=self.centered_screen_size//2
        self.validation_passed=True
        self.fix_game_data()
        self.global_map=np.full(tuple(self.global_map_shapes.tolist()),no_map_tile_id,dtype=np.uint8,order="C")
        self.global_map=fill_pad_image(self.global_map,*self.global_map_padding,reserve_first_channels=True,fill_value=no_map_tile_id)
        self.call_direct_script("script_core_set_start_positions")
        self.last_checkpoint_place=self.first_checkpoint_place.copy()
        #self.no_assets=True
        should_load_gfx=not self.no_assets and (self.use_gfx_image or self.from_gui) and "global_starting_coords" in self.game_data
        self.tile_id_colors_lookup=np.array(tile_id_colors_list,dtype=np.uint8,order="C")
        self.global_map_gfx=self.game_selector.fallback_read_map_png() if should_load_gfx else np.zeros((16,16,3),dtype=np.uint8,order="C")
        self.sprites_gfx=self.game_selector.fallback_read_sprites_png() if should_load_gfx else np.zeros((16,16,3),dtype=np.uint8,order="C")
        self.characters_gfx=generate_characters_tiles(self.scaled_sprites_size,self.scaled_sprites_size,downscale=1,fill_value=0xFF) if should_load_gfx else np.zeros((256,16,9,3),dtype=np.uint8,order="C")
        self.global_map_gfx_padding=self.tile_size*(self.centered_screen_size//2)
        self.tile_color_players=np.array([self.tile_id_colors_lookup[player_down_tile_id].tolist(),[0x9F,0x4F,0x4F]],dtype=np.uint8,order="C")
        if grayscale_screen:
            self.tile_color_players=rgb_to_grayscale(self.tile_color_players,shrink_axis=grayscale_single_channel)
            self.tile_id_colors_lookup=rgb_to_grayscale(self.tile_id_colors_lookup,shrink_axis=grayscale_single_channel)
            self.characters_gfx=rgb_to_grayscale(self.characters_gfx,shrink_axis=grayscale_single_channel,equal_split=True)
            self.global_map_gfx=rgb_to_grayscale(self.global_map_gfx,shrink_axis=grayscale_single_channel)
            self.sprites_gfx=rgb_to_grayscale(self.sprites_gfx,shrink_axis=grayscale_single_channel)
        if should_load_gfx and (np.prod(np.array(self.global_map_gfx.shape[:2]))<160000) or (np.prod(np.array(self.sprites_gfx.shape[:2]))<4096):
            self.use_gfx_image=False
            self.screen_observation_type=max(0,min(3,self.screen_observation_type))
            self.gfx_loaded=False
        else:
            self.gfx_loaded=True
        if not self.from_gui and not self.log_screen:
            self.hsv_image=False
        self.global_map_gfx=fill_pad_image(self.global_map_gfx,*self.global_map_gfx_padding,reserve_first_channels=False,fill_value=0x00)
        if self.screen_downscale>1:
            if grayscale_screen:
                self.sprites_gfx[self.sprites_gfx>0xFB]=0xFF
            self.global_map_gfx=downscale_local_mean(self.global_map_gfx,(self.screen_downscale,self.screen_downscale,1)).astype(np.uint8)
            self.sprites_gfx=downscale_local_mean(self.sprites_gfx,(self.screen_downscale,self.screen_downscale,1)).astype(np.uint8)
            self.global_map_gfx_padding//=self.screen_downscale
        if grayscale_screen:
            self.sprites_gfx[self.sprites_gfx>0xFB]=0xFF
        self.sprites_gfx=tile_split(self.sprites_gfx,self.scaled_sprites_size)
        self.walkable_tiles=np.array(walkable_tiles_ids,dtype=np.uint8)
        self.npc_walkable_tiles=np.array(list(set(walkable_tiles_ids)-set([bush_tile_id,warp_tile_id])),dtype=np.uint8)
        self.automatic_scripted_map_ids=np.array([],dtype=np.uint8)
        self.define_internal_data()
        self.call_direct_script("script_core_set_automatic_map_ids")
        self.action_nop_id=self.movement_max_actions
        self.action_interact_id=self.action_nop_id+(1 if self.button_interaction and self.action_nop else 0)
        self.action_back_id=self.action_interact_id+1
        self.action_menu_id=self.action_interact_id+2
        self.action_extra_id=self.action_interact_id+3
        self.action_menu_max_id=self.action_extra_id
        defined_actions_powerups_ids_count=self.define_actions_ids()
        extra_action_space=0
        if self.action_nop:
            extra_action_space+=1
        if self.button_interaction:
            extra_action_space+=1
            if self.true_menu:
                extra_action_space+=3
            elif not self.bypass_powerup_actions:
                extra_action_space+=defined_actions_powerups_ids_count
        self.secondary_action_value=0
        self.forced_directional_movements=[]
        self.reward_range=(0,15000)
        self.all_actions_count=self.movement_max_actions+extra_action_space
        screen_channels_list=[] if grayscale_screen else [3]
        if self.screen_observation_type==0:
            (self.screen_box_high,self.screen_box_shape)=(1,(1,1))
        elif self.screen_observation_type==1:
            (self.screen_box_high,self.screen_box_shape)=(count_tiles_ids-1,self.centered_screen_size.tolist())
        elif self.screen_observation_type==2:
            (self.screen_box_high,self.screen_box_shape)=(0x01,self.centered_screen_size.tolist()+[count_tiles_ids])
        elif self.screen_observation_type==3:
            (self.screen_box_high,self.screen_box_shape)=(0xFF,self.centered_screen_size.tolist()+screen_channels_list)
        elif self.screen_observation_type==4:
            (self.screen_box_high,self.screen_box_shape)=(0xFF,(self.centered_screen_size*self.scaled_sprites_size).tolist()+screen_channels_list)
        else:
            (self.screen_box_high,self.screen_box_shape)=(count_tiles_ids-1,self.centered_screen_size.tolist())
        self.teleport_data={}
        self.event_rewards_powerups=deepcopy(self.game_data["powerups"])
        self.event_rewards_data=OrderedDict(deepcopy(self.filter_dict_cache(self.game_data["events"])))
        self.scripts_data=deepcopy(self.filter_dict_cache(self.game_data["scripts"]))
        self.npcs_data=deepcopy(self.game_data["npcs"]) if self.using_npcs else {}
        self.event_rewards_data_by_map={}
        self.scripts_data_by_map={}
        self.npcs_data_by_map={}
        self.npcs_map_lookup={k:v[0][0] for k,v in self.npcs_data.items()}
        self.last_npc_name=""
        self.extra_event_names=[]
        self.event_flags_lookup={}
        self.first_event_name=""
        self.trigger_done_event_name=""
        self.define_game_critical_event_names()
        self.call_direct_script("script_core_set_extra_event_names")
        self.reset_event_flags(False)
        self.infinite_game=bool(config.get("infinite_game",False))
        self.sandbox=bool(config.get("sandbox",False))
        self.define_game_config(config)
        self.history_tracking={
            "visited_maps":np.zeros_like(self.global_map,dtype=np.uint8,order="C"),
            "steps_on_map":np.zeros_like(self.global_map,dtype=np.uint32,order="C"),
        }
        self.game_state={
            "player_coordinates_data":np.zeros((11,),np.int16,order="C"),
            "last_suggested_coordinates":np.zeros((4,),np.int16,order="C"),
            "previous_map_id":self.first_checkpoint_place[0],
            "powerup_walk_tile":0,"powerup_screen_remove_tile":0,"powerup_screen_fix_tile":0,"powerup_started":0,
            "player_level":0,"loss_count":0,"pseudo_seed":0,
            "menu_type":0,"sub_menu_type":0,"text_type":0,"battle_type":0,
            "event_flags":np.zeros((len(self.event_flags_lookup),),dtype=np.uint8,order="C"),
            "game_minimal_screen":np.zeros(self.centered_screen_size.tolist(),dtype=np.uint8,order="C"),
        }
        self.stacked_state={}
        self.game_state.update(self.define_extra_game_state())
        self.stacked_state.update(self.define_extra_stacked_state())
        self.define_game_config_post_game_state_creation(config)
        if agent_class is not None:
###         self.load_custom_save_from_events(self.starting_event,self.starting_collected_flags)
            self.agent=agent_class(self,**{**kwargs,**agent_args})
            self.agent.set_env(self)
        self.hook_init()
        self.reset_game()
        self.build_global_connections()
        self.fix_npc_game_data()
#####################
### FIX GAME DATA ###
#####################
    def is_valid_game_data(self,raise_exception:bool=True,verbose:bool=True)->bool:
        """Check if game_data parameters have possible inconsistencies."""
        ret=True
        global_shapes_correct=True
        data_shapes={k:self.game_data[k].shape[0] for k in ["sizes","channels","legacy_global_starting_coords"]}#"warps"
        if len(set(data_shapes.values()))>1:
            ret=False
            global_shapes_correct=False
            exc_text=f"\tLenght of most relevant map information data differs: {data_shapes}."
            if verbose:
                print(exc_text)
            if raise_exception:
                raise IndexError(exc_text)
            return False
        data_ranges=[0x1F,0x200,0x4000]
        maps_count=self.get_maps_count()
        if self.game_data["channels"].min()<0 or self.game_data["channels"].max()>data_ranges[0]:
            global_shapes_correct=False
            if verbose:
                ret=False
                print(f"\tMap [channels] should be in range [0,{data_ranges[0]:d}).")
            else:
                return False
        if self.game_data["sizes"].min()<0 or self.game_data["sizes"].max()>data_ranges[1]:
            global_shapes_correct=False
            if verbose:
                ret=False
                print(f"\tMap [sizes] should be in range [1,{data_ranges[1]:d}).")
            else:
                return False
        if self.game_data["legacy_global_starting_coords"].min()<0 or self.game_data["legacy_global_starting_coords"].max()>data_ranges[2]:
            global_shapes_correct=False
            if verbose:
                ret=False
                print(f"\tMap [legacy_global_starting_coords] should be in in range [0,{data_ranges[2]:d}).")
            else:
                return False
        for map_id in range(self.game_data["sizes"].shape[0]):
            if map_id not in self.game_data["maps"]:
                if verbose:
                    ret=False
                    print(f"\tMap [0x{map_id:03X} : {map_id:d}] has sizes but no map data.")
                else:
                    return False
        for map_id,v in self.game_data["maps"].items():
            if map_id>self.game_data["sizes"].shape[0]:
                if verbose:
                    ret=False
                    print(f"\tMap [sizes: {map_id:d} ('0x{map_id:03X}')] not declared.")
                else:
                    return False
            elif not np.array_equal(v.shape,self.game_data["sizes"][map_id]):
                if verbose:
                    ret=False
                    print(f"\tMap [sizes: {map_id:d} ('0x{map_id:03X}')] discrepancy. Expected: {self.game_data['sizes'][map_id]} Declared: {v.shape}")
                else:
                    return False
        for map_id in range(self.game_data["warps"].shape[0]):
            (warps_finished,warps_after_empty)=(-1,False)
            for warp_id,warp in enumerate(self.game_data["warps"][map_id]):
                empty_warp=False
                if np.min(warp)<0:
                    if verbose:
                        ret=False
                        print(f"\tWarp [{warp_id:d}] of map [{map_id:d} ('0x{map_id:03X}')] uses negative values.")
                    else:
                        return False
                elif np.max(warp)==0:
                    empty_warp=True
                elif warp[2]>=maps_count:
                    if verbose:
                        ret=False
                        print(f"\tWarp [{warp_id:d}] of map [{map_id:d} ('0x{map_id:03X}')] targets a map [{warp[2]:d} ('0x{warp[2]:03X}')] that doesn't exist.")
                    else:
                        return False
                if empty_warp:
                    if warps_finished<0:
                        warps_finished=warp_id
                elif warps_finished>=0:
                    warps_after_empty=True
            if warps_after_empty:
                if verbose:
                    ret=False
                    print(f"\tMap [{map_id:d} ('0x{map_id:03X}')] has warps declared after position [{warps_finished}] which is an empty array.")
                else:
                    return False
        if global_shapes_correct:
            temp_shapes=np.empty((3,),dtype=np.int16,order="C")
            temp_shapes[0]=np.max(self.game_data["channels"].clip(0,0x1F))+1
            temp_shapes[1:3]=0x200+self.game_data["legacy_global_starting_coords"].clip(0,0x4000).max(axis=0)
            temp_matrix=np.zeros(tuple(temp_shapes.tolist()),dtype=np.uint32,order="C")
            for map_id in range(maps_count):
                channel=np.clip(self.game_data["channels"][map_id],0,0x1F)
                dims=[np.clip(self.game_data["legacy_global_starting_coords"][map_id],0,0x4000),
                    np.clip(self.game_data["sizes"][map_id],1,0x200)]
                bounds=np.hstack([dims[0],dims[0]+dims[1]],dtype=np.int16)
                if np.min(dims)>1:
                    temp_matrix[channel,bounds[0]:bounds[2],bounds[1]:bounds[3]]+=1
            overlaps=np.where(temp_matrix>1,1,0).sum()
            if temp_matrix.max()>1:
                if verbose:
                    ret=False
                    print(f"\tDifferent maps are overlapping tiles [{overlaps:d}]! You should adjust this by editing the fields [channels, sizes, legacy_global_starting_coords].")
                    self.show_map_matrix(np.where(temp_matrix[0]>1,2,0))
                else:
                    return False
        return ret
    def is_valid_npc_game_data(self,raise_exception:bool=True,verbose:bool=True)->bool:
        """Check if npcs game_data parameters have possible inconsistencies."""
        if len(self.npcs_data_by_map)==0:
            return True
        ret=True
        for map_id,npcs_data in self.npcs_data_by_map.items():
            map_coords_set=set()
            for idx,npc in npcs_data.items():
                enc_coords=npc[0][1]*100000+npc[0][2]
                if enc_coords in map_coords_set:
                    if verbose:
                        ret=False
                        print(f"\tMap [{map_id:d} ('0x{map_id:03X}')] npc [{idx}] positioned on another npc.")
                    else:
                        return False
                map_coords_set.add(enc_coords)
                if npc[3]==warp_tile_id:
                    if verbose:
                        ret=False
                        print(f"\tMap [{map_id:d} ('0x{map_id:03X}')] npc [{idx}] positioned on a warp.")
                    else:
                        return False
        return ret
    def fix_game_data(self)->None:
        """Fix game_data parameters for possible inconsistencies. Doesn't fix critical errors."""
        if not self.skip_validation:
            self.validation_passed=self.is_valid_game_data(raise_exception=True,verbose=True)
        self.game_data["sizes"][:]=self.game_data["sizes"].clip(0,0x200)
        for map_id,map_size in enumerate(self.game_data["sizes"]):
            if map_id not in self.game_data["maps"]:
                self.game_data["maps"][map_id]=np.full(map_size,walk_tile_id,dtype=np.int16,order="C")
        maps_count=self.get_maps_count()
        data_shapes={k:self.game_data[k].shape[0] for k in ["sizes","channels","legacy_global_starting_coords","warps"]}
        if len(set(data_shapes.values()))>1:
            for k,v in data_shapes.items():
                if k=="sizes":
                    continue
                min_v=min(v,data_shapes["sizes"])
                temp_val=np.zeros(tuple([data_shapes["sizes"]]+list(self.game_data[k].shape[1:])),dtype=self.game_data[k].dtype,order="C")
                temp_val[:min_v]=self.game_data[k][:min_v]
                self.game_data[k]=temp_val
        self.game_data["channels"][:]=self.game_data["channels"].clip(0,0x1F)
        self.game_data["legacy_global_starting_coords"][:]=self.game_data["legacy_global_starting_coords"].clip(0,0x4000)
        del_map_ids=[map_id for map_id,_ in self.game_data["maps"].items() if map_id<0 or map_id>=maps_count]
        for map_id in del_map_ids:
            del self.game_data["maps"][map_id]
        if not self.padding_fixed:
            self.game_data["legacy_global_starting_coords"]+=self.global_map_padding
            self.padding_fixed=True
        if "bounds" not in self.game_data:
            self.game_data["bounds"]=np.zeros((maps_count,4),dtype=np.int16,order="C")
        for map_id,map_tiles in self.game_data["maps"].items():
            map_start=self.game_data["legacy_global_starting_coords"][map_id]
            map_dims=self.game_data["sizes"][map_id]
            assert len(map_dims)>1 and all(map_start>=self.global_map_padding) and np.max(map_dims)>=0
            if len(self.game_data["maps"][map_id].shape)<2:
                self.game_data["maps"][map_id]=np.full(tuple(map_dims.tolist()),fill_value=walk_tile_id,dtype=map_tiles.dtype,order="C")
            elif not np.array_equal(map_tiles.shape[:2],map_dims):
                min_v=np.vstack([np.array(map_tiles.shape[:2],dtype=map_dims.dtype,order="C"),map_dims],dtype=map_dims.dtype).min(axis=0)
                temp_val=np.full(tuple(map_dims.tolist()),fill_value=walk_tile_id,dtype=map_tiles.dtype,order="C")
                temp_val[:min_v[0],:min_v[1]]=map_tiles[:min_v[0],:min_v[1]]
                self.game_data["maps"][map_id]=temp_val
            self.game_data["bounds"][map_id]=np.hstack([map_start,map_start+map_dims],dtype=np.int16)
        self.global_map_shapes[0]=np.max(self.game_data["channels"])+1
        self.global_map_shapes[1:3]=(self.game_data["bounds"][:,2:4].max(axis=0)-self.global_map_padding).clip(0,0x4200)
        if "connections_mask" not in self.game_data:
            self.game_data["connections_mask"]=np.zeros((maps_count,4),dtype=np.uint8,order="C")
        if "warps_count" not in self.game_data:
            self.game_data["warps_count"]=np.zeros((maps_count,),dtype=np.uint8,order="C")
        for map_id,map_warps in enumerate(self.game_data["warps"]):
            warps_count=0
            invalid_warps=[]
            for warp_id,warp in enumerate(map_warps):
                if np.min(warp)<0 or warp[2]>=maps_count:
                    warps_count+=1
                    invalid_warps.append(warp_id)
                elif np.max(warp)==0:
                    break
                else:
                    warps_count+=1
                    try:
                        self.game_data["maps"][map_id][warp[0],warp[1]]=warp_tile_id
                    except (KeyError,IndexError):
                        invalid_warps.append(warp_id)
            if len(invalid_warps)>0:
                mask=np.repeat(np.expand_dims(np.hstack([
                    np.isin(np.arange(warps_count,dtype=np.uint16),invalid_warps),
                    np.ones(map_warps.shape[0]-warps_count,dtype=bool)
                ],dtype=bool),axis=1),map_warps.shape[1],axis=1)
                self.game_data["warps"][map_id][:]=np.pad(map_warps[~mask],(0,mask.sum()),constant_values=0).reshape(mask.shape)
            warps_count-=len(invalid_warps)
            try:
                self.game_data["warps_count"][map_id]=warps_count
                self.game_data["warps"][map_id][warps_count:]=0
            except (KeyError,IndexError):
                pass
    def fix_npc_game_data(self)->None:
        """Fix npcs game_data parameters for possible inconsistencies. Doesn't fix critical errors."""
        #if not self.skip_validation:
        self.validation_passed=self.is_valid_npc_game_data(raise_exception=True,verbose=True)
#################################
### GAME SPECIFIC DEFINITIONS ###
#################################
    def get_menu_class(self)->Any:
        """Return the class of the menu object."""
        return None
    def game_enforce_config(self,config:dict)->dict:
        """Alter the configurations dict to enforce specific options."""
        return config
    def define_game_tile_size(self)->int:
        """Sprite/Tile size in pixels."""
        return 16
    def define_game_screen_tiles_size(self)->tuple:
        """Screen size in tiles unit."""
        return (9,9)
    def define_game_extra_data_functions(self)->list:
        """List of extra game data files functions."""
        return []
    def define_extra_required_game_data_names(self)->list:
        """List of required keys that must exist in the dictionary data."""
        return []
    def define_game_config(self,config:dict)->None:
        """Game-specific configurations initialization."""
        return
    def define_game_config_post_game_state_creation(self,config:dict)->None:
        """Game-specific configurations initialization run after game state declaration."""
    def define_internal_data(self)->None:
        """Game-specific attribute declarations."""
        return
    def define_actions_ids(self)->int:
        """Custom game actions id declaration."""
        return 0
    def define_extra_game_state(self)->dict:
        """Dict of extra values preserved in the game_state dictionary."""
        return {}
    def define_extra_stacked_state(self)->dict:
        """Dict of extra stacked values preserved in the stacked_state dictionary."""
        return {}
#########################
### GENERIC-UTILITIES ###
#########################
    def check_recursion(self,limit:int=1)->bool:
        """Check if recursion depth is not above the limit."""
        return self.step_recursion_depth<=limit
    def filter_dict_cache(self,cache:dict)->dict:
        """Strip dict key if they are not supposed to be used in NPC mode."""
        return filter_dict_key_depending_on_npc(cache,self.using_npcs)
    def deepcopy(self)->Any:
        """Deepcopy of the environment, avoiding pickle fails due to module objects."""
        stripped_attrs={}
        for k in ["scripting_core_scope"]:
            if hasattr(self,k):
                stripped_attrs[k]=getattr(self,k)
                setattr(self,k,None)
        new_env=deepcopy(self)
        for k,v in stripped_attrs.items():
            setattr(self,k,v)
            setattr(new_env,k,v)
        return new_env
    def clear_single_lru_cache(self,func_name:str)->bool:
        """Clear the caching function of a method if properly decorated."""
        if hasattr(self,func_name):
            func=getattr(self,func_name)
            if hasattr(func,"cache_clear") and callable(func.cache_clear):
                func.cache_clear()
                return True
        return False
    def clear_all_lru_cache(self)->None:
        """Clear any cache decorator."""
        for k in dir(self):
            attr=getattr(self,k)
            if callable(attr):
                self.clear_single_lru_cache(k)
##################
### VALIDATION ###
##################
    def get_game_name(self,add_custom:bool=False)->str:
        """Returns the game name from the selector."""
        return f"{self.game_selector.game_name}{'' if not add_custom or not self.game_selector.is_custom_game else ' - Custom'}"
    def validator_reinitialize(self)->None:
        """Custom fixed reinitialization for validation purposes."""
        config=deepcopy(self.env_config)
        config.update({"action_complexity":3,"benchmarking":True,"skip_validation":True})
        self.env_config.clear()
        self.game_data.clear()
        self.__init__(config)
###############################
### USER-FRIENDLY FUNCTIONS ###
###############################
    def get_current_map_id(self)->np.ndarray:
        """Current map id."""
        return self.game_state["player_coordinates_data"][0]
    def get_current_channel_id(self)->np.ndarray:
        """Current map channel id."""
        return self.game_state["player_coordinates_data"][4]
    def get_current_legacy_global_map_player_channel_and_coords(self)->np.ndarray:
        """Return legacy global coordinates [channel,l_global_y,l_global_x]."""
        return self.game_state["player_coordinates_data"][4:7]
    def get_current_legacy_global_player_coordinates(self)->np.ndarray:
        """Return legacy global coordinates [l_global_y,l_global_x]."""
        return self.game_state["player_coordinates_data"][5:7]
    def get_last_legacy_global_channel_zero_player_coords(self)->np.ndarray:
        """Return last channel_0 legacy global coordinates [lc0_global_y,lc0_global_x]."""
        return self.game_state["player_coordinates_data"][7:9]
    def get_current_global_player_coordinates(self)->np.ndarray:
        """Return last global coordinates [global_y,global_x]."""
        return self.game_state["player_coordinates_data"][9:11]
#############
### AGENT ###
#############
    def has_agent(self)->bool:
        """Return if any agent is linked."""
        return self.agent is not None
    def set_agent(self,agent,recursion_depth:int=0)->None:
        """Link an agent to the game."""
        self.agent=agent
        if recursion_depth<2 and hasattr(self.agent,"set_env"):
            self.agent.set_env(self,recursion_depth+1)
##################
### GAME HOOKS ###
##################
    def hook_init(self)->None:
        """Game Hook: executed at game initialization."""
        return
    def hook_reset(self)->None:
        """Game Hook: executed at game reset."""
        return
    def hook_before_warp(self,global_warped:bool,movements:list)->None:
        """Game Hook: executed before entering a warp."""
        return
    def hook_after_warp(self)->None:
        """Game Hook: executed after exiting a warp."""
        return
    def hook_after_movement(self)->None:
        """Game Hook: executed after moving a tile."""
        return
    def hook_after_step(self,action:int=-1)->None:
        """Game Hook: executed at the end of the game step."""
        return
    def hook_after_script(self,key:str,script_data:list,should_delete_script:bool=False)->None:
        """Game Hook: executed after a script runs."""
        return
    def hook_after_event(self,key:str,event_data:list,use_script_positions:bool=False)->None:
        """Game Hook: executed after an event is activated."""
        return
    def hook_update_overworld_screen(self)->None:
        """Game Hook: executed updating the screen while in overworld."""
        return
###################
### DEBUG HOOKS ###
###################
    def hook_get_debug_text(self)->str:
        """Extra text printed in gui mode for debug purposes."""
        return ""
###############
### SEEDING ###
###############
    def set_game_seed(self,seed:Union[int,None]=None)->None:
        """Set the same seed on most random number generators."""
        self.seed=seed
    def pseudo_random_npc_movement_chance(self,off:int=0)->bool:
        """Faster pseudo-random chance. Return True to move all NPC."""
        return (self.step_count//256+self.game_state["player_coordinates_data"][0]+self.game_state["pseudo_seed"]+off)%4==3
##########################
### DATA-MAP FUNCTIONS ###
##########################
    def get_maps_count(self)->int:
        """Return the amount of maps in game_data."""
        return len(self.game_data["sizes"])
    @lru_cache_func(maxsize=2048)
    def get_cached_map_sizes(self,map_id:int)->np.ndarray:
        """Return map sizes."""
        return self.game_data["sizes"][map_id][:2]
    @lru_cache_func(maxsize=2048)
    def get_cached_map_connections_mask(self,map_id:int)->int:
        """Return map connections masks."""
        return self.game_data["connections_mask"][map_id][:4]
    @lru_cache_func(maxsize=2048)
    def get_cached_legacy_global_map_starting_coords(self,map_id:int)->np.ndarray:
        """Return map legacy starting global coordinates."""
        return self.game_data["legacy_global_starting_coords"][map_id][:2]
    @lru_cache_func(maxsize=2048)
    def get_cached_global_map_starting_coords(self,map_id:int)->np.ndarray:
        """Return map starting global coordinates or legacy if none is found."""
        return self.game_data["global_starting_coords"][map_id][:2] if "global_starting_coords" in self.game_data else self.get_cached_legacy_global_map_starting_coords(map_id)
    @lru_cache_func(maxsize=2048)
    def get_cached_global_map_channel(self,map_id:int)->int:
        """Return map channel id."""
        return self.game_data["channels"][map_id]
    @lru_cache_func(maxsize=2048)
    def get_cached_global_map_bounds(self,map_id:int)->int:
        """Return map rect bounds."""
        return self.game_data["bounds"][map_id][:4]
    @lru_cache_func(maxsize=2048)
    def get_cached_global_map_area(self,map_id:int)->int:
        """Return product of map shape."""
        return np.prod(self.game_data["sizes"][map_id][:2])
    @lru_cache_func(maxsize=2048)
    def get_cached_map_warps(self,map_id:int)->np.ndarray:
        """Return map warps."""
        return self.game_data["warps"][map_id][:self.game_data["warps_count"][map_id]] if "warps" in self.game_data else self.game_data["objects"][map_id]["warps"]
    @lru_cache_func(maxsize=2048)
    def get_cached_map_npcs(self,map_id:int)->np.ndarray:
        """Return map NPC."""
        return self.game_data["npcs"][map_id][:self.game_data["npcs_count"][map_id]] if "npcs" in self.game_data else self.game_data["objects"][map_id]["npcs"]
    def get_legacy_global_map_coords(self,map_id:int,y:int,x:int)->np.ndarray:
        """Return legacy global coordinates given local data."""
        return self.get_cached_legacy_global_map_starting_coords(map_id)+np.array([y,x],dtype=np.int16)
    @lru_cache_func(maxsize=65536)
    def inner_get_cached_legacy_global_map_coords(self,map_id:int,y:int,x:int)->np.ndarray:
        """Return legacy global coordinates given local data."""
        return self.get_cached_legacy_global_map_starting_coords(map_id)+np.array([y,x],dtype=np.int16)
    def get_cached_legacy_global_map_coords(self,map_id:int,local_coords:np.ndarray)->np.ndarray:
        """Return legacy global coordinates given local data."""
        return self.inner_get_cached_legacy_global_map_coords(map_id,*local_coords[:2])
    def get_global_map_coords(self,map_id:int,y:int,x:int)->np.ndarray:
        """Return global coordinates given local data."""
        return self.get_cached_global_map_starting_coords(map_id)+np.array([y,x],dtype=np.int16)
    @lru_cache_func(maxsize=65536)
    def inner_get_cached_global_map_coords(self,map_id:int,y:int,x:int)->np.ndarray:
        """Return global coordinates given local data."""
        return self.get_cached_global_map_starting_coords(map_id)+np.array([y,x],dtype=np.int16)
    def get_cached_global_map_coords(self,map_id:int,local_coords:np.ndarray)->np.ndarray:
        """Return global coordinates given local data."""
        return self.inner_get_cached_global_map_coords(map_id,*local_coords[:2])
    @lru_cache_func(maxsize=65536)
    def cached_is_point_inside_area(self,yp:int,xp:int,y1:int,x1:int,y2:int,x2:int)->bool:
        """Return if a point is inside a rect."""
        return y2>yp>=y1 and x2>xp>=x1
#        return nb_is_point_inside_area(yp,xp,y1,x1,y2,x2)
    @lru_cache_func(maxsize=4096)
    def cached_check_point_bounds_map(self,map_id:int,y:int,x:int)->int:
        """Return the map_id belonging to a point. Used for global map transitions."""
        for tmap,bounds in (self.game_data["bounds"].items() if isinstance(self.game_data["bounds"],dict) else enumerate(self.game_data["bounds"])):
            if self.cached_is_point_inside_area(y,x,*bounds):
                return tmap
        return map_id
##########################
### DIRECTION HANDLING ###
##########################
    @lru_cache_func(maxsize=8)
    def get_action_to_direction(self,action:int,placeholder:int=0)->int:
        """Convert an action into a direction."""
        return placeholder if action<0 else self.action_directions[action]
    @lru_cache_func(maxsize=5)
    def get_inverted_direction(self,direction:int)->int:
        """Invert the direction value."""
        return self.inverted_directions[direction]
    @lru_cache_func(maxsize=5)
    def get_direction_offset(self,direction:int)->np.ndarray:
        """Convert a direction into [y,x] offsets from player position."""
        return self.directions_offsets[direction]
    @lru_cache_func(maxsize=5)
    def get_direction_from_offsets_4way(self,y:int,x:int)->np.ndarray:
        """Convert [y,x] offsets into a direction with movement_max_actions=4."""
        for direction,offs in enumerate(self.directions_offsets):
            if y==offs[0] and x==offs[1]:
                return direction
        return 4
    @lru_cache_func(maxsize=5)
    def get_action_from_direction_offsets_4way(self,y:int,x:int)->np.ndarray:
        """Convert [y,x] offsets into an action with movement_max_actions=4."""
        direction=self.get_direction_from_offsets_4way(y,x)
        for action,act_dir in enumerate(self.action_directions):
            if direction==act_dir:
                return action
        return 0
    @lru_cache_func(maxsize=5)
    def get_action_offset(self,movement:int)->np.ndarray:
        """Convert an action info [y,x] offset from player position."""
        return self.movement_offsets[movement]
    def get_faced_direction_coordinates(self,y:int,x:int,direction:int=4)->np.ndarray:
        """Get local coordinates of a point facing a particular direction."""
        return np.array([y,x],dtype=np.int16)+self.get_direction_offset(direction)
    @lru_cache_func(maxsize=128)
    def get_expected_direction_and_movement(self,max_movements:int,action:int,current_direction:int)->np.ndarray:
        """Main convertion of actions into directional data."""
        ret=np.empty((3,),dtype=np.int16,order="C")
        if action>=max_movements:
            ret[2]=current_direction
            ret[:2]=self.get_direction_offset(ret[2])
        elif max_movements<4:
            if action<=0:
                ret[2]=current_direction
            else:
                action=min(action,max_movements-1)
                if action==1:
                    ret[2]=np.roll(self.action_directions,-1)[current_direction]
                else:
                    ret[2]=(np.roll(self.action_directions,-3)[current_direction]+2)%4
            ret[:2]=self.get_direction_offset(ret[2])
        else:
            ret[:2]=self.get_action_offset(action)
            ret[2]=self.get_action_to_direction(action,current_direction)
        return ret
########################
### FORCED MOVEMENTS ###
########################
    def reset_forced_directional_movements(self)->None:
        """Clear all forced movements."""
        self.forced_directional_movements.clear()
    def add_forced_action(self,action:int)->None:
        """Add a forced action to the queue. Used to set powerup actions from menu."""
        self.forced_directional_movements.append(action)
    def add_forced_directional_movements(self,direction:int)->None:
        """Add a forced directional movement to the queue."""
        self.forced_directional_movements.append(direction)
    def add_forced_movement_down(self)->None:
        """Add a forced down movement to the queue."""
        self.add_forced_directional_movements(0)
    def add_forced_movement_left(self)->None:
        """Add a forced left movement to the queue."""
        self.add_forced_directional_movements(2)
    def add_forced_movement_right(self)->None:
        """Add a forced right movement to the queue."""
        self.add_forced_directional_movements(3)
    def add_forced_movement_up(self)->None:
        """Add a forced up movement to the queue."""
        self.add_forced_directional_movements(1)
    def add_forced_movement_invert_direction(self)->None:
        """Add a forced direction inversion movement to the queue."""
        self.add_forced_directional_movements(self.get_inverted_direction(self.game_state["player_coordinates_data"][3]))
#############
### TILES ###
#############
    def fix_tile(self,tile_id:int)->int:
        """Fix the tile value depending on the powerup state."""
        return (self.game_state["powerup_screen_fix_tile"] if self.game_state["powerup_screen_fix_tile"]>walk_tile_id else walk_tile_id) if self.game_state["powerup_screen_remove_tile"]>0 and tile_id==self.game_state["powerup_screen_remove_tile"] else tile_id
    def get_current_tile(self,fix:bool=True)->int:
        """Tile at player coordinates."""
        tile=self.global_map[self.game_state["player_coordinates_data"][4],self.game_state["player_coordinates_data"][5],self.game_state["player_coordinates_data"][6]]
        return self.fix_tile(tile) if fix else tile
    def get_faced_tile(self,fix:bool=True)->int:
        """Tile faced by the player."""
        global_faced_coords=self.game_state["player_coordinates_data"][4:7].copy()
        global_faced_coords[1:3]+=self.get_direction_offset(self.game_state["player_coordinates_data"][3])
#        nb_sum_int16(global_faced_coords[1:3],self.get_direction_offset(self.game_state["player_coordinates_data"][3]))
        tile=self.global_map[global_faced_coords[0],global_faced_coords[1],global_faced_coords[2]]
        return self.fix_tile(tile) if fix else tile
    @lru_cache_func(maxsize=1024)
    def is_walkable_tile_id(self,tile_id:int,powerup_walk_tile:int=0)->bool:
        """Return True if the tile can be traversed."""
        return tile_id in self.walkable_tiles or (powerup_walk_tile>0 and powerup_walk_tile==tile_id)
#        return nb_in_array_uint8(tile_id,self.walkable_tiles) or (powerup_walk_tile>0 and powerup_walk_tile==tile_id)
    @lru_cache_func(maxsize=512)
    def is_npc_walkable_tile_id(self,tile_id:int,in_water:bool=False)->bool:
        """Return True if the tile can be traversed by NPC."""
        return tile_id in self.npc_walkable_tiles or (in_water and tile_id==water_tile_id)
#        return nb_in_array_uint8(tile_id,self.npc_walkable_tiles) or (in_water and tile_id==water_tile_id)
    def get_game_powerup_tiles_dict(self)->dict:
        """Return a dict with powerup keys and walkable tile values."""
        return {}
    @lru_cache_func(maxsize=256)
    def is_powerup_walkable_tile_id(self,tile_id:int)->bool:
        """Return True if the tile can be ever traversed by future powerups."""
        for _,tiles_list in self.get_game_powerup_tiles_dict().items():
            for powerup_walk_tile in (tiles_list if isinstance(tiles_list,(list,set,tuple)) else [tiles_list]):
                if powerup_walk_tile>1 and self.is_walkable_tile_id(tile_id,powerup_walk_tile):
                    return True
        return False
    def special_tile_direction_check(self,tile_id:int,direction:int)->bool:
        """Return True if the tile can be traversed."""
        (force_movement,valid,move_direction,movement_count)=self.inner_special_tile_direction_check(tile_id,direction)
        if force_movement:
            self.reset_forced_directional_movements()
            if movement_count>0:
                self.forced_directional_movements+=[move_direction for _ in range(movement_count)]
        return valid
    @lru_cache_func(maxsize=1024)
    def inner_special_tile_direction_check(self,tile_id:int,direction:int)->tuple[bool,bool,int]:
        """Internal check for tile movement validation."""
        valid_jump_directions={ledge_down_tile_id:0,ledge_up_tile_id:1,
            ledge_left_tile_id:2,ledge_right_tile_id:3}
        valid_move_directions={indoor_puzzle_platform_reset_tile_id:0,
            indoor_puzzle_platform_down_tile_id:0,water_current_down_tile_id:0,
            indoor_puzzle_platform_up_tile_id:1,water_current_up_tile_id:1,
            indoor_puzzle_platform_left_tile_id:2,water_current_left_tile_id:2,
            indoor_puzzle_platform_right_tile_id:3,water_current_right_tile_id:3,
        }
        valid_movements_count={
            ledge_down_tile_id:1,ledge_up_tile_id:1,
            ledge_left_tile_id:1,ledge_right_tile_id:1,
            indoor_puzzle_platform_reset_tile_id:0,
            indoor_puzzle_platform_down_tile_id:8,water_current_down_tile_id:1,
            indoor_puzzle_platform_up_tile_id:8,water_current_up_tile_id:1,
            indoor_puzzle_platform_left_tile_id:8,water_current_left_tile_id:1,
            indoor_puzzle_platform_right_tile_id:8,water_current_right_tile_id:1,
        }
        tile_jump_direction=valid_jump_directions.get(tile_id,4)
        tile_move_direction=valid_move_directions.get(tile_id,4)
        force_movement=tile_jump_direction==direction or tile_move_direction<4
        move_direction=direction if tile_jump_direction<4 else tile_move_direction
        movement_count=valid_movements_count.get(tile_id,1)
        return (force_movement,force_movement or tile_jump_direction>=4,move_direction,movement_count)
############
### MENU ###
############
    def has_menu(self)->bool:
        """Return if any menu is linked."""
        return self.menu is not None
    def set_menu(self,menu,recursion_depth:int=0)->None:
        """Link an menu to the game."""
        self.menu=menu
        if recursion_depth<2 and hasattr(self.menu,"set_env"):
            self.menu.set_env(self,recursion_depth+1)
    def clear_text(self)->None:
        """Clears the text queue."""
        if len(self.text_queue)>0:
            self.text_queue.clear()
            self.text_changed_during_step=True
    def set_text(self,text:str=None,append:bool=False,max_line_size:int=16)->None:
        """Correctly set or append text to the queue."""
        self.text_changed_during_step=True
        if not self.true_menu or text is None:
            self.game_state["text_type"]=1
        else:
            if not append:
                self.clear_text()
            new_queue=[k[i:i+max_line_size] for k in text.split("\n") for i in range(0,len(k),max_line_size)]
            self.text_queue.extend(new_queue)
    def step_text(self,max_lines:int=2,single_line:bool=True,keep_end:bool=False)->bool:
        """Slice the displayed text at the bottom of the screen."""
        self.text_changed_during_step=True
        if len(self.text_queue)>max_lines:
            self.text_queue=self.text_queue[1 if single_line else max_lines:]
        else:
            if not keep_end:
                self.clear_text()
            return True
        return False
    def set_placeholder_npc_text(self)->None:
        """Set placeholder text for NPC."""
        if self.has_menu() and len(self.text_queue)==0:
            self.clear_text()
            self.menu.set_npc_text_with_presses_count("NPC text.")
    def show_npc_text(self):
        """Prepare settings to display NPC text."""
        if self.game_state["menu_type"]==0:
            if self.has_menu():
                self.menu.return_to_text_menu()
            else:
                self.game_state["text_type"]=1
            self.set_placeholder_npc_text()
    def close_menu(self):
        """Close any menu."""
        if self.has_menu():
            self.menu.return_to_overworld()
        self.game_state["menu_type"]=0
        self.game_state["text_type"]=0
    def clear_menu_content(self,until_depth=-1,sticky_text:bool=False):
        """Clear any menu string data until a depth."""
        if self.menu_current_depth<0:
            return
        self.menu_changed_during_step=True
        for i in ([self.menu_current_depth] if until_depth<0 else range(self.menu_current_depth,until_depth-1,-1)):
            self.menu_content_by_depth[i].clear()
            self.menu_bg_by_depth[i,:]=0
            self.menu_cursor_data[i,:]=0
        if not sticky_text:
            self.clear_text()
        self.menu_current_depth=max(0,until_depth)-1
    def append_menu_content(self,y_tile_pos:int,x_tile_pos:int,text:str=0):
        """Append string data at the current given position."""
        self.menu_content_by_depth[self.menu_current_depth].append([y_tile_pos,x_tile_pos,text])
    def append_multiple_menu_contents(self,y_tile_pos:int,x_tile_pos:int,vertical:bool=True,text_list:list=None,clear_content:bool=False,sticky_text:bool=True):
        """Append a list of strings data continously."""
        if clear_content and self.menu_current_depth>=0:
            self.menu_content_by_depth[self.menu_current_depth].clear()
        if not sticky_text:
            self.clear_text()
        if text_list is None:
            return
        self.menu_changed_during_step=True
        if vertical:
            for i,k in enumerate(text_list):
                self.env.append_menu_content(y_tile_pos+i,x_tile_pos,k)
        else:
            for i,k in enumerate(text_list):
                self.env.append_menu_content(y_tile_pos,x_tile_pos+i,k)
    def increment_menu_depth(self,y1:int,x1:int,y2:int,x2:int,displayed:bool=True)->None:
        """Increment the menu depth and set the bg area."""
        self.menu_changed_during_step=True
        self.menu_current_depth=min(7,self.menu_current_depth+1)
        vals=[y1,x1,y2,x2,displayed]
        for i in range(0,3,2):
            if vals[i]<0:
                vals[i]+=self.centered_screen_size[0]
        for i in range(1,4,2):
            if vals[i]<0:
                vals[i]+=self.centered_screen_size[1]
        self.menu_bg_by_depth[self.menu_current_depth,:]=vals
    def set_menu_cursor_origin(self,y_tile_pos:int,x_tile_pos:int,vertical:bool=True,displayed:bool=True,value:int=0)->None:
        """Set the origin position of a cursor."""
        self.menu_cursor_data[self.menu_current_depth,:4]=[y_tile_pos,x_tile_pos,vertical,displayed]
        self.set_menu_cursor_value(value)
    def set_menu_cursor_value(self,value:int)->None:
        """Set the current position of a cursor relative to its origin."""
        self.menu_cursor_data[self.menu_current_depth,4]=value
        self.menu_cursor_data[self.menu_current_depth,5:7]=self.menu_cursor_data[self.menu_current_depth,:2]
        self.menu_cursor_data[self.menu_current_depth,5 if self.menu_cursor_data[self.menu_current_depth,2]>0 else 6]+=value
    def set_new_menu_layer(self,y1:int,x1:int,y2:int,x2:int,displayed:bool=True,
        y_cursor:int=0,x_cursor:int=0,vertical:bool=True,displayed_cursor:bool=True,value:int=0,
    sticky_text:bool=False,clear_until_depth:int=-2)->None:
        """Increment the menu depth and set background and cursor values."""
        if clear_until_depth>-2:
            self.clear_menu_content(clear_until_depth,sticky_text)
        self.increment_menu_depth(y1,x1,y2,x2,displayed)
        self.set_menu_cursor_origin(y_cursor,x_cursor,vertical,displayed_cursor,value)
    @lru_cache_func(maxsize=65536)
    def encode_text_to_tiles(self,txt:str)->np.ndarray:
        """Convert text to tiles_id. Each tile encodes 2 characters."""
        if len(txt)%2==1:
            txt+=" "
        enc_characters=np.take(self.encoded_characters_lookup,np.array(struct.unpack(f"{len(txt):d}B",bytes(txt,"ascii")),dtype=np.uint8,order="C"),axis=0)
        enc_tiles=0xFF-enc_characters[::2]-len(self.shrinked_characters_list)*enc_characters[1::2]
        return enc_tiles
############################
### GLOBAL-MAP FUNCTIONS ###
############################
    def change_global_map_tile_from_faced_local_coords(self,map_id:int,y:int,x:int,direction:int,tile_id:int,check_npc:bool=True)->int:
        """Changes the global map tile at faced coordinates. Used to replace reward and script tiles."""
        channel=self.get_cached_global_map_channel(map_id)
        coords=self.get_legacy_global_map_coords(map_id,*self.get_faced_direction_coordinates(y,x,direction))
        old_tile=self.global_map[channel,coords[0],coords[1]]
        global_pos_fallback=True
        if check_npc:
            # and old_tile in [npc_down_tile_id,npc_left_tile_id,npc_right_tile_id,npc_up_tile_id]:
            lcoords=np.array([y,x],dtype=np.int16)+self.get_direction_offset(direction)
            for v in self.npcs_data_by_map.get(map_id,{}).values():
                if np.array_equal(v[1][0,:2],lcoords):
                    global_pos_fallback=False
                    v[3]=tile_id
                    v[4]=1
                    break
        if global_pos_fallback:
            self.global_map[channel,coords[0],coords[1]]=tile_id
        return old_tile
    def change_global_map_tile_from_local_coords(self,map_id:int,y:int,x:int,tile_id:int,check_npc:bool=True)->int:
        """Changes the global map tile at faced coordinates. Used to replace reward and script tiles."""
        return self.change_global_map_tile_from_faced_local_coords(map_id,y,x,4,tile_id,check_npc)
    def get_map_data_from_local_coordinates(self,map_id:int,y:int=0,x:int=0)->np.ndarray:
        """Get map data at any local location."""
        map_data=np.empty((len(self.game_state["player_coordinates_data"]),),dtype=self.game_state["player_coordinates_data"].dtype,order="C")
        if not isinstance(map_id,np.ndarray):
            map_data[:3]=np.array([map_id,y,x],dtype=map_data.dtype)
#            nb_assign_int16(map_data[:3],np.array([map_id,y,x],dtype=map_data.dtype))
        else:
            map_data[:3]=map_id[:3]
#            nb_assign_int16(map_data[:3],map_id[:3])
        map_data[4]=self.get_cached_global_map_channel(map_data[0])
        map_data[5:7]=self.get_cached_legacy_global_map_coords(map_data[0],map_data[1:3])
#        nb_assign_int16(map_data[5:7],self.get_cached_legacy_global_map_coords(map_data[0],map_data[1:3]))
        if map_data[4]==0:
            map_data[7:9]=map_data[5:7]
#            nb_assign_int16(map_data[7:9],map_data[5:7])
        map_data[9:11]=self.get_cached_global_map_coords(map_data[0],map_data[1:3])
#        nb_assign_int16(map_data[9:11],self.get_cached_global_map_coords(map_data[0],map_data[1:3]))
        return map_data
    def transfer_scripting_tiles(self,src_map:np.ndarray,dest_map:np.ndarray)->np.ndarray:
        """Transfer particular tiles id from one map to another. Used internally."""
        mask=np.isin(src_map,[active_script_tile_id,old_script_tile_id,active_reward_tile_id,old_reward_tile_id,npc_down_tile_id,npc_left_tile_id,npc_right_tile_id,npc_up_tile_id])
        dest_map[mask]=src_map[mask]
        return dest_map
    def build_global_single_map_without_scripting(self,map_id:int,copy_scripting_tiles:bool)->None:
        """Allocate a single map in the global map."""
        cur_map_start=self.game_data["legacy_global_starting_coords"][map_id]
        cur_map_dims=self.game_data["sizes"][map_id]
        if len(cur_map_dims)>1 and np.max(cur_map_start)>=0 and np.max(cur_map_dims)>=0:
            if copy_scripting_tiles:
                new_map=self.game_data["maps"][map_id].copy()
                new_map=self.transfer_scripting_tiles(self.global_map[self.game_data["channels"][map_id],cur_map_start[0]:cur_map_start[0]+cur_map_dims[0],cur_map_start[1]:cur_map_start[1]+cur_map_dims[1]],new_map)
                self.global_map[self.game_data["channels"][map_id],cur_map_start[0]:cur_map_start[0]+cur_map_dims[0],cur_map_start[1]:cur_map_start[1]+cur_map_dims[1]]=new_map
            else:
                self.global_map[self.game_data["channels"][map_id],cur_map_start[0]:cur_map_start[0]+cur_map_dims[0],cur_map_start[1]:cur_map_start[1]+cur_map_dims[1]]=self.game_data["maps"][map_id]
    def reindex_current_map_warps(self,map_id:int)->None:
        """Change temporarily all map warps of a given map id. It can be used for elevators."""
### TO IMPLEMENT
### FUNCTION TO EDIT MAP WARPS TO NEW LOCATION. IT MAY NEED RECACHING THE DATA. UNKNOWN DEFAULT VALUES
        self.clear_single_lru_cache("get_cached_map_warps")
    def apply_global_scripting_tiles(self,map_changed:bool=False)->None:
        """Transfer particular tiles to all maps. Used internally."""
        for v in self.scripts_data.values():
            self.change_global_map_tile_from_local_coords(*v[0][:3],active_script_tile_id,False)
        if not self.using_npcs:
            for v in self.event_rewards_data.values():
                self.change_global_map_tile_from_faced_local_coords(*v[0][:4],active_reward_tile_id,False)
        for map_id,_ in self.npcs_data_by_map.items():
            self.npcs_map_tiles_update(map_id,map_changed)
    def build_global_map(self,skip_scripting:bool=False)->None:
        """Build all the global map."""
        if not skip_scripting:
            self.apply_global_scripting_tiles(map_changed=False)
        for (cur_map_id,_) in self.game_data["maps"].items():
            self.build_global_single_map_without_scripting(cur_map_id,False)
        if self.strip_ledges:
            self.global_map[np.isin(self.global_map,[ledge_down_tile_id,ledge_up_tile_id])]=walk_tile_id
            self.global_map[np.isin(self.global_map,[ledge_left_tile_id,ledge_right_tile_id])]=unwalkable_tile_id
        if not skip_scripting:
            self.apply_global_scripting_tiles(map_changed=True)
    def build_global_connections(self)->None:
        """Builds the walkable connection points of the various maps at channel 0."""
        ch0_maps=[[map_id,self.get_cached_global_map_bounds(map_id),self.get_cached_map_sizes(map_id)] for map_id in range(self.get_maps_count()) if self.get_cached_global_map_channel(map_id)==0 and self.get_cached_map_connections_mask(map_id).sum()>0 and np.prod(self.get_cached_map_sizes(map_id))>=4]
        self.connection_points={m[0]:[[] for i in range(4)] for m in ch0_maps}
        for md in ch0_maps:
            ranges=[
                [md[1][1],md[1][3],md[1][2]-1,[1,0]],
                [md[1][0],md[1][2],md[1][1],[0,-1]],
                [md[1][0],md[1][2],md[1][3],[0,1]],
                [md[1][1],md[1][3],md[1][0],[-1,0]],
            ]
            for d,r in enumerate(ranges):
                for c in range(r[0],r[1]):
                    coords=np.array([[r[2],c],[r[2],c]],dtype=np.int16)
                    if r[3][0]==0:
                        coords=np.flip(coords,axis=1)
                    coords[1]+=np.array(r[3],dtype=np.int16)
                    if self.is_powerup_walkable_tile_id(self.global_map[0,coords[0][0],coords[0][1]]) and self.is_powerup_walkable_tile_id(self.global_map[0,coords[1][0],coords[1][1]]):
                        mid=self.cached_check_point_bounds_map(md[0],*coords[1])
                        if mid!=md[0]:
                            self.connection_points[md[0]][d].append(np.hstack([[mid],coords[1].tolist(),coords[0].tolist()],dtype=np.int16))
######################
### EVENT-HANDLING ###
######################
    def set_start_positions(self,start:list,checkpoint:list)->None:
        """Player start position and checkpoint on a fresh start."""
        self.start_map_position=np.array((start+[0]*4)[:4],dtype=np.int16).clip(0,None)
        self.first_checkpoint_place=np.array((checkpoint+[0]*4)[:4],dtype=np.int16).clip(0,None)
    def define_game_critical_event_names(self)->None:
        """Define first and last events of the game."""
        self.first_event_name=""
        self.trigger_done_event_name=""
    def set_extra_event_names(self,event_names:list)->None:
        """Extra events non-declared via files data."""
        self.extra_event_names=[k for k in event_names if k not in self.event_rewards_data]
    def reset_event_flags(self,update_state:bool=True)->None:
        """Initialize all flags and their name lookup."""
        all_events=list(self.event_rewards_data.keys())
        all_events+=list(sorted(set([self.first_event_name,self.trigger_done_event_name]+list(self.event_rewards_powerups.keys())+[f"first_{k}" for k in self.event_rewards_powerups.keys()]+list(r for ev in self.event_rewards_powerups.values() for r in ev)+list(self.extra_event_names))-set(all_events)))
        event_flags=OrderedDict([[k,0] for k in all_events if len(k)>0])
        self.game_completed=False
        self.event_flags_lookup={k:i for i,(k,_) in enumerate(event_flags.items())}
        if update_state and "event_flags" in self.game_state:
            self.game_state["event_flags"][:]=0
    @lru_cache_func(maxsize=4096)
    def get_event_flag(self,name:str)->int:
        """Return the value of a given event flag."""
        idx=self.event_flags_lookup.get(name,-1)
        return self.game_state["event_flags"][idx] if idx>=0 else 0
    @lru_cache_func(maxsize=1)
    def get_event_flags_sum(self)->int:
        """Return the sum of all event flags."""
        return np.ndarray.sum(self.game_state["event_flags"])
    def clear_events_caches(self)->bool:
        """Clear cache decorators of event-related functions."""
        ret=True
        for k in ["get_event_flag","get_event_flags_sum"]:
            if not self.clear_single_lru_cache(k):
                ret=False
        return ret
    def update_powerups_flags(self)->bool:
        """Update powerup flags depending on their prerequisites."""
        for k,v in self.event_rewards_powerups.items():
            self.game_state["event_flags"][self.event_flags_lookup.get(k,0)]=1 if len([1 for rk in v if self.get_event_flag(rk)!=1])==0 else 0
        return self.clear_events_caches()
    def game_on_event_flag_change(self,name:str,activated:bool)->bool:
        """Used to for validating event states. Return True to clear again the event cache."""
        return False
    def game_on_event_custom_load(self,starting_event:str,used_collected_flags:set,used_level:int)->None:
        """Game-specific custom state load fixes."""
        return
    def deactivate_event_flag(self,name:str)->bool:
        """Set an event flag to 0."""
        if name not in self.event_flags_lookup:
            return False
        self.game_state["event_flags"][self.event_flags_lookup.get(name,0)]=0
        self.clear_events_caches()
        if self.game_on_event_flag_change(name,False):
            self.clear_events_caches()
        self.update_powerups_flags()
        return True
    def activate_event_flag(self,name:str)->bool:
        """Set an event flag to 1. Stops the game if the event is the last one."""
        if name not in self.event_flags_lookup:
            return False
        self.game_state["event_flags"][self.event_flags_lookup.get(name,0)]=1
        self.clear_events_caches()
        if self.game_on_event_flag_change(name,True):
            self.clear_events_caches()
        self.update_powerups_flags()
        if len(self.trigger_done_event_name)>0 and name==self.trigger_done_event_name:
            self.game_completed=True
        return True
    def activate_event_reward_flag(self,name:str,use_script_positions:bool=False)->bool:
        """Set an event flag to 1 and update tiles or NPC states."""
        if name not in self.event_rewards_data or self.get_event_flag(name)>0:
            return False
        self.activate_event_flag(name)
        if not self.using_npcs:
            self.change_global_map_tile_from_faced_local_coords(*self.event_rewards_data[name][0][:4],old_reward_tile_id,True)
        self.hook_after_event(name,self.event_rewards_data[name],use_script_positions)
        return True
    def get_active_flags_names(self)->list[str]:
        """Gets the name of all active flags."""
        return [k for v,(k,_) in zip(self.game_state["event_flags"],self.event_flags_lookup.items()) if v>0]
###########################
### GAME-STATE HANDLING ###
###########################
    def roll_assign_stacked_state_by_name(self,name:str,obj:np.ndarray)->None:
        """Assign a new value to the stacked-state object, rolling back the previous values."""
        if name not in self.stacked_state or not isinstance(self.stacked_state[name],np.ndarray):
            return
        self.stacked_state[name]=np.roll(self.stacked_state[name],1,axis=0)
        if len(self.stacked_state[name].shape)>2:
            self.stacked_state[name][0]=obj
#            nb_assign_int16_2d(self.stacked_state[name][0],obj)
        else:
            self.stacked_state[name][0]=obj
#            nb_assign_int16(self.stacked_state[name][0],obj)
####################
### NPC HANDLING ###
####################
    def update_single_npc_data_tiles(self,npc_data,map_changed:bool)->list:
        """Handles proper assign of NPC tile and overridden map tile."""
        if npc_data[3]>=0 and not map_changed and npc_data[4]==0:
            # or np.array_equal(npc_data[1][0],npc_data[1][1])):continue
            return npc_data
        if npc_data[3]>0:
            self.change_global_map_tile_from_local_coords(npc_data[0][0],*npc_data[1][1,:2],npc_data[3],False)
        if map_changed:
            npc_data[1][0]=npc_data[0][1:4]
#            nb_assign_int16(npc_data[1][0],npc_data[0][1:4])
        if npc_data[0][5]==1:
            npc_data[3]=self.change_global_map_tile_from_local_coords(npc_data[0][0],*npc_data[1][0,:2],npc_down_tile_id+npc_data[1][0,2],False)
        npc_data[1][1]=npc_data[1][0]
#        nb_assign_int16(npc_data[1][1],npc_data[1][0])
        npc_data[4]=0
        return npc_data
    def update_npcs_movements_and_interactions(self,interactions:bool=True)->str:
        """Main routine for NPC movements and interaction."""
        npc_script_name=None
        npc_script_args=[]
        check_movement=interactions and self.check_recursion(1)
        check_interaction=check_movement and self.last_action==self.action_interact_id and self.game_state["text_type"]==0 and self.game_state["battle_type"]==0
        if check_interaction:
            faced_coords=self.get_faced_direction_coordinates(*self.game_state["player_coordinates_data"][1:4])
        check_movement&=self.pseudo_random_npc_movement_chance()
        map_dims=self.get_cached_map_sizes(self.game_state["player_coordinates_data"][0]) if check_movement else []
        map_channel=self.get_cached_global_map_channel(self.game_state["player_coordinates_data"][0]) if check_movement else 0
        for npc_name,v in self.npcs_data_by_map.get(self.game_state["player_coordinates_data"][0],{}).items():
            should_update=(v[0][5]>0)&(v[0][7]>0)&check_movement
            if v[0][5]>0:
                if check_interaction and np.array_equal(v[1][0,:2],faced_coords) and (v[0][5]!=3 or self.game_state["player_coordinates_data"][3]==1):
                    self.last_npc_name=npc_name
                    v[5]+=1
                    v[1][0,2]=self.get_inverted_direction(self.game_state["player_coordinates_data"][3])
                    if isinstance(v[2],str):
                        npc_script_name=v[2]
                    elif isinstance(v[2],(list,set,tuple)) and len(v[2])>0:
                        npc_script_name=v[2][0]
                        if len(v[2])>1:
                            npc_script_args=v[2][1:]
                    check_interaction=False
                    should_update=True
                elif should_update:
#                    if v[0][7]==1:
#                        v[1][0,2]=np.random.randint(0,4)
#                    elif v[0][7]==2:
#                        v[1][0,2]=np.random.randint(0,2)
#                    elif v[0][7]==3:
#                        v[1][0,2]=2+np.random.randint(0,2)
#                    elif v[0][7]>=4 and v[0][7]<=7:
#                        v[1][0,2]=v[1][0,2]-4
#                    else:
#                        v[1][0,2]=np.random.randint(0,4)
                    if v[0][6]>0:
                        new_expected_coords=v[1][0,:2].copy()
                        new_expected_coords+=self.get_direction_offset(v[1][0,2])
#                        nb_sum_int16(new_expected_coords,self.get_direction_offset(v[1][0,2]))
### HANDLE MOVING ONLY IF THE SPRITE IS INSIDE THE COLLISIONS!
                        if self.cached_is_point_inside_area(*new_expected_coords,0,0,*map_dims):
                            new_expected_global_coords=self.get_cached_legacy_global_map_coords(self.game_state["player_coordinates_data"][0],new_expected_coords)
                            expected_tile=self.global_map[map_channel,new_expected_global_coords[0],new_expected_global_coords[1]]
#                           print(v[1][0,:2],new_expected_coords,expected_tile)
                            if self.is_npc_walkable_tile_id(expected_tile):
#                               print(expected_tile)
                                v[1][0,:2]+=self.get_direction_offset(v[1][0,2])
            if v[4]<1:
                v[4]=1 if should_update else 0
            v=self.update_single_npc_data_tiles(v,False)
        return (npc_script_name,npc_script_args)
    def npcs_map_tiles_update(self,map_id:int,map_changed:bool)->None:
        """Update all NPC tiles."""
        for v in self.npcs_data_by_map.get(map_id,{}).values():
            v=self.update_single_npc_data_tiles(v,map_changed)
    def get_dummy_npc_configs(self,relative:bool=False)->dict:
        """Placeholder for NPC configs."""
        return {"y":0,"x":0,"direction":0,"sprite":8,"state":0,"relative":relative,"interactions":0,"won":0,"exists":False}
    def get_npc_configs(self,npc_name:str,relative:bool=False)->dict:
        """Return the current configs of the NPC."""
        map_id=self.npcs_map_lookup.get(npc_name,-1)
        if map_id>=0 and npc_name in self.npcs_data_by_map.get(map_id,{}):
            rel=bool(relative)
            v=self.npcs_data_by_map[map_id][npc_name]
            config=dict(zip(["y","x","directon"],v[1][0,:3]))
### TO-IMPLEMENT: RELATIVE COORDINATES RETURN
###         if rel:pass
            config.update({"sprite":v[0][4],"state":v[0][5],"relative":rel,"interactions":v[5],"won":v[6],"exists":True})
            return config
        return self.get_dummy_npc_configs()
    def modify_npc_configs(self,npc_name:str,config:dict,permanent:bool=True,force_update:bool=True)->None:
        """Edit NPC configs."""
        map_id=self.npcs_map_lookup.get(npc_name,-1)
        if map_id<0 or npc_name not in self.npcs_data_by_map.get(map_id,{}):
            return self.get_dummy_npc_configs()
        if force_update:
            self.npcs_map_tiles_update(map_id,False)
        v=self.npcs_data_by_map[map_id][npc_name]
        rel=config.get("relative",False)
        neutral_coords=np.zeros((2,),dtype=np.int16,order="C") if rel else (v[1][0,:2] if map_id==self.game_state["player_coordinates_data"][0] else v[0][1:3])
        used_config={"y":neutral_coords[0],"x":neutral_coords[1],"direction":-1,"sprite":-1,"state":-1,"relative":rel,"interactions":-1,"won":-1}
        used_config.update(config)
        new_pos=np.array([used_config[k] for k in ["y","x"]],dtype=np.int16)
        if rel:
            v[1][0,:2]+=new_pos
        else:
            v[1][0,:2]=new_pos
        if used_config["direction"]>=0:
            v[1][0,2]=int(used_config["direction"])%4
        if used_config["sprite"]>=0:
            v[0][4]=int(used_config["sprite"])
        if used_config["state"]>=0:
            v[0][5]=int(used_config["state"])%3
        if used_config["interactions"]>=0:
            v[5]=int(used_config["interactions"])
        if used_config["won"]>=0:
            v[6]=int(used_config["won"]>0)
        if permanent:
            v[0][1:3]=v[1][0,:2]
        v[4]=1
        if force_update:
            self.npcs_map_tiles_update(map_id,False)
            if self.game_state["battle_type"]==0:
                self.update_overworld_screen()
        used_config.update({"interactions":v[5],"won":v[6]})
        return used_config
########################
### SCREEN-RENDERING ###
########################
    def update_overworld_screen(self)->None:
        """Draw the overworld screen."""
        if not self.totally_headless:
            bounds=self.game_state["player_coordinates_data"][5:7].repeat(2)+self.player_screen_bounds
            self.game_state["game_minimal_screen"][:]=self.global_map[self.game_state["player_coordinates_data"][4],bounds[0]:bounds[1],bounds[2]:bounds[3]]
#            nb_assign_uint8_2d(self.game_state["game_minimal_screen"],self.global_map[self.game_state["player_coordinates_data"][4],bounds[0]:bounds[1],bounds[2]:bounds[3]])
            self.history_tracking["visited_maps"][self.game_state["player_coordinates_data"][4],bounds[0]:bounds[1],bounds[2]:bounds[3]]=self.game_state["game_minimal_screen"]
#            nb_assign_uint8_2d(self.history_tracking["visited_maps"][self.game_state["player_coordinates_data"][4],bounds[0]:bounds[1],bounds[2]:bounds[3]],self.game_state["game_minimal_screen"])
            self.game_state["game_minimal_screen"][self.player_screen_position[0],self.player_screen_position[1]]=player_down_tile_id+self.game_state["player_coordinates_data"][3]
            if self.game_state["powerup_screen_remove_tile"]>0:
                off=self.player_screen_position+self.get_direction_offset(self.game_state["player_coordinates_data"][3])
                self.game_state["game_minimal_screen"][off[0],off[1]]=self.fix_tile(self.game_state["game_minimal_screen"][off[0],off[1]])
        self.hook_update_overworld_screen()
    @lru_cache_func(maxsize=1024)
    def get_string_position(self,y_pos:int,x_pos:int,use_characters:bool)->np.ndarray:
        """Return the real start position of the string."""
        pos=np.array([y_pos,x_pos,0,0],dtype=np.int16,order="C")
        for i in range(2):
            if pos[i]<0:
                pos[i]+=self.centered_screen_size[i]
        if use_characters:
            pos*=self.scaled_sprites_size
        pos[2:4]=pos[:2]
        return pos
    def draw_tile_string(self,y_pos:int,x_pos:int,text:str)->None:
        """Draw the encoded text tiles screen."""
        pos=self.get_string_position(y_pos,x_pos,False)
        chr_tiles=self.encode_text_to_tiles(text)
        try:
            if pos[1]+len(chr_tiles)>=self.centered_screen_size[1]:
                raise ValueError()
            self.game_state["game_minimal_screen"][pos[0],pos[1]:pos[1]+len(chr_tiles)]=chr_tiles
        except (ValueError,IndexError):
            for i,tile in enumerate(chr_tiles):
                try:
                    self.game_state["game_minimal_screen"][pos[0],pos[1]+i]=tile
                except IndexError:
                    break
    def update_text_screen(self)->None:
        """Draw the text screen."""
        if not self.totally_headless and self.screen_observation_type<4:
            self.game_state["game_minimal_screen"][-3:]=generic_menu_tile_id
            self.set_placeholder_npc_text()
            if len(self.text_queue)>0:
                self.draw_tile_string(-2,1,self.text_queue[0])
    def update_menu_screen(self)->None:
        """Draw the menu screen."""
        if self.totally_headless:
            return
        if not self.true_menu:
            self.game_state["game_minimal_screen"][:,-5:]=generic_menu_tile_id
            return
        if self.menu_current_depth<0:
            if self.game_state["text_type"]>0:
                self.update_text_screen()
            return
        if self.screen_observation_type>=4:
            return
        if self.menu_current_depth>=0:
            shapes_differ=len(self.last_gfx_menu_overlap)<2 or not np.array_equal(self.last_gfx_menu_overlap.shape,self.game_state["game_minimal_screen"].shape)
            if self.menu_changed_during_step or shapes_differ:
                for i in range(self.menu_current_depth+1):
                    bg=self.menu_bg_by_depth[i]
                    if bg[4]>0:
                        self.game_state["game_minimal_screen"][bg[0]:bg[2],bg[1]:bg[3]]=generic_menu_tile_id
                    for mc in self.menu_content_by_depth[i]:
                        self.draw_tile_string(*mc[:3])
                if shapes_differ:
                    self.last_gfx_menu_overlap=self.game_state["game_minimal_screen"].copy()
                else:
                    self.last_gfx_menu_overlap[:]=self.game_state["game_minimal_screen"]
            else:
                self.game_state["game_minimal_screen"][:]=self.last_gfx_menu_overlap
            if self.menu_cursor_data[self.menu_current_depth,3]>0:
                self.game_state["game_minimal_screen"][self.menu_cursor_data[self.menu_current_depth,5],self.menu_cursor_data[self.menu_current_depth,6]]=generic_cursor_tile_id
        if self.game_state["text_type"]>0:
            self.update_text_screen()
    def draw_monocromatic_player(self,img:np.ndarray,upscale:int=1)->np.ndarray:
        """Draw the player in monocromatic mode."""
        img[upscale*self.player_screen_position[0]:upscale*self.player_screen_position[0]+upscale,upscale*self.player_screen_position[1]:upscale*self.player_screen_position[1]+upscale]=self.tile_color_players[0]
        if upscale>1:
            off=upscale*self.player_screen_position+self.get_direction_offset(self.game_state["player_coordinates_data"][3])+upscale//2
            img[off[0],off[1]]=self.tile_color_players[1]
        return img
    def get_monocromatic_agent_screen(self,upscale:int=1,draw_player_direction:bool=True)->np.ndarray:
        """Draw the screen in monocromatic mode."""
        img=map_matrix_to_image(self.game_state["game_minimal_screen"],self.tile_id_colors_lookup)
        if upscale>1:
            img=np.repeat(np.repeat(img,upscale,axis=0),upscale,axis=1)
            if draw_player_direction and self.game_state["menu_type"]<2:
                img=self.draw_monocromatic_player(img,upscale)
        return img
    def draw_gfx_npc(self,img:np.ndarray,sprite_id:int,direction:int,position:np.ndarray,upscale:int=1,apply_tile_offset:bool=False)->np.ndarray:
        """Draw an NPC from preloaded sprites."""
        character=self.sprites_gfx[sprite_id,direction]
        if upscale>1:
            character=np.repeat(np.repeat(character,upscale,axis=0),upscale,axis=1)
        bounds=position.repeat(2)
        bounds[1::2]+=1
        bounds*=self.scaled_sprites_size*upscale
        if apply_tile_offset:
            bounds[:2]=(bounds[:2]-(self.scaled_sprites_size*upscale)//4).clip(0)
            ch_offs=self.scaled_sprites_size-np.diff(bounds.reshape(2,2),axis=1).flatten()
            character=character[ch_offs[0]:,ch_offs[1]:]
        img[bounds[0]:bounds[1],bounds[2]:bounds[3]]=character if self.hsv_image else np.where(character==0xFF,img[bounds[0]:bounds[1],bounds[2]:bounds[3]],character)
        return img
    def draw_gfx_player(self,img:np.ndarray,upscale:int=1)->np.ndarray:
        """Draw the player from preloaded sprites."""
        return self.draw_gfx_npc(img,0,self.game_state["player_coordinates_data"][3],self.player_screen_position,upscale)
    def draw_gfx_string(self,img:np.ndarray,y_tile_pos:int,x_tile_pos:int,text:str,upscale:int=1)->np.ndarray:
        """Draw the a string on screen from preloaded characters."""
        pos=self.get_string_position(y_tile_pos,x_tile_pos,True)
        idx_list=[]
        for character in text:
            idx=ord(character)
            idx_list.append(idx)
            if idx in {10,13}:
                break
        try:
            img[pos[2]:pos[2]+self.characters_gfx.shape[1],pos[3]:pos[3]+len(idx_list)*self.characters_gfx.shape[2]]=np.hstack([self.characters_gfx[idx] for idx in idx_list],dtype=np.uint8)
        except ValueError:
            pos=pos.copy()
            for idx in idx_list:
                try:
                    img[pos[2]:pos[2]+self.characters_gfx.shape[1],pos[3]:pos[3]+self.characters_gfx.shape[2]]=self.characters_gfx[idx]
                except ValueError:
                    break
                if idx in {10,13}:
                    pos[2]+=self.characters_gfx.shape[1]
                    pos[3]=pos[1]
                else:
                    pos[3]+=self.characters_gfx.shape[2]
        return img
    def get_gfx_overworld_screen_view(self,upscale:int=1)->np.ndarray:
        """Return the view of the current gfx screen."""
        bounds=self.scaled_sprites_size*(self.game_state["player_coordinates_data"][9:11].repeat(2)+self.player_screen_bounds)+self.global_map_gfx_padding.repeat(2)
        return self.global_map_gfx[bounds[0]:bounds[1],bounds[2]:bounds[3]]
    def draw_gfx_text(self,img:np.ndarray)->np.ndarray:
        """Draw the text."""
        if len(self.text_queue)==0 and self.game_state["text_type"]==0:#or not self.text_changed_during_step
            return img
        img[-3*self.scaled_sprites_size:]=0xFF
        for i,v in enumerate(self.text_queue[:2]):
            img=self.draw_gfx_string(img,-2+i,1,v)
        return img
    def draw_gfx_menu(self,img:np.ndarray)->np.ndarray:
        """Draw the menu."""
        if self.menu_current_depth>=0:
            shapes_differ=img is None or not np.array_equal(self.last_gfx_menu_overlap.shape,img.shape)
            if (self.menu_changed_during_step and shapes_differ) or len(self.last_gfx_menu_overlap.shape)<2:
                if len(self.last_gfx_menu_overlap.shape)<2:
                    self.last_gfx_menu_overlap=self.get_gfx_overworld_screen_view().copy() if img is None else img.copy()
                else:
                    self.last_gfx_menu_overlap[:]=self.get_gfx_overworld_screen_view() if img is None else img
            if self.menu_changed_during_step:
                for i in range(self.menu_current_depth+1):
                    bg=self.menu_bg_by_depth[i]
                    if bg[4]>0:
                        bg=bg.copy()*self.scaled_sprites_size
                        self.last_gfx_menu_overlap[bg[0]:bg[2],bg[1]:bg[3]]=0xFF-i*0x1F
                    for mc in self.menu_content_by_depth[i]:
                        self.last_gfx_menu_overlap=self.draw_gfx_string(self.last_gfx_menu_overlap,*mc[:3])
                img=self.last_gfx_menu_overlap.copy()
            elif img is None or shapes_differ:
                img=self.last_gfx_menu_overlap.copy()
            else:
                img[:]=self.last_gfx_menu_overlap
            if self.menu_cursor_data[self.menu_current_depth,3]>0:
                img=self.draw_gfx_string(img,*self.menu_cursor_data[self.menu_current_depth,5:7],">")
        elif img is None:
            img=self.get_gfx_overworld_screen_view().copy()
        img=self.draw_gfx_text(img)
        return img
    def get_gfx_agent_screen(self,upscale:int=1)->np.ndarray:
        """Draw the screen from preloaded tiles."""
        if self.game_state["menu_type"] in {0,2}:
            img=self.get_gfx_overworld_screen_view().copy()
            npc_max_coords=self.centered_screen_size if self.game_state["text_type"]==0 else self.centered_screen_size-np.array([3,0],dtype=np.int16)
            for v in self.npcs_data_by_map.get(self.game_state["player_coordinates_data"][0],{}).values():
                if v[0][5]!=1:
                    continue
                rel_pos=v[1][0,:2]-self.game_state["player_coordinates_data"][1:3]+self.player_screen_position
                if self.cached_is_point_inside_area(*rel_pos,0,0,*npc_max_coords):
                    img=self.draw_gfx_npc(img,v[0][4],v[1][0,2],rel_pos)
            img=self.draw_gfx_player(img)
            if self.hsv_image:
                mono_img=np.full(img.shape,0xFF,dtype=img.dtype,order="C")
                mono_img=self.get_monocromatic_agent_screen(self.scaled_sprites_size,draw_player_direction=False)
                img=rgb_to_hsv(img)
                mono_img=rgb_to_hsv(mono_img)
                for ax,alpha in enumerate([0.25,0.75,0.5]):
                    img[:,:,ax]=alpha*img[:,:,ax]+(1.-alpha)*mono_img[:,:,ax]
                img=hsv_to_rgb(img)
                img=img.astype(np.uint8)
        else:
            img=None
        if self.game_state["menu_type"]>0:
            img=self.draw_gfx_menu(img)
        elif self.game_state["text_type"]>0 and self.game_state["battle_type"]==0:
            img=self.draw_gfx_text(img)
        return img
    def get_agent_screen(self,upscale:int=1)->np.ndarray:
        """Return the agent screen."""
        return self.get_gfx_agent_screen(upscale) if self.use_gfx_image and not self.totally_headless else self.get_monocromatic_agent_screen(upscale)
    def screen_ndarray(self)->np.ndarray:
        """Return the agent screen at predefined scale."""
        return self.get_agent_screen(self.default_monocromatic_screen_scale)
    def get_map_minimal_screen(self,from_history:bool=False,pad:int=1,relative:bool=False)->np.ndarray:
        """Return a view of the minimal screen."""
        if relative:
            return self.get_monocromatic_agent_screen()
        cur_map_id=self.game_state["player_coordinates_data"][0]
        bounds=self.get_cached_global_map_bounds(cur_map_id)
        return (self.history_tracking["visited_maps"] if from_history else self.global_map)[self.get_cached_global_map_channel(cur_map_id),bounds[0]-pad:bounds[2]+pad,bounds[1]-pad:bounds[3]+pad]
##################
### GIF-SAVING ###
##################
    def reset_gif_frames(self)->None:
        """Clear all saved gif frames."""
        self.gif_frames.clear()
    def add_gif_frame(self)->None:
        """Log a new screen to the gif list."""
        if hasattr(self,"game_state"):
            self.gif_frames.append(self.get_agent_screen(5))
    def save_gif(self,outfile_or_buff:Union[str,BytesIO,None]=None,return_buff:int=True,delete_old:bool=True,speedup:int=4,loop:bool=False)->Union[bool,BytesIO]:
        """Builds the gif and save it to a file or buffer."""
        if speedup<1:
            used_speedup=1 if len(self.gif_frames)<200 else 4
        else:
            used_speedup=int(used_speedup)
        for _ in range((4*used_speedup)-1):
            self.add_gif_frame()
        ret=generate_gif_from_numpy(self.gif_frames,outfile_or_buff,return_buff,1000*24/60./used_speedup,loop)
        if delete_old:
            self.reset_gif_frames()
        return ret
    def save_run_gif(self,delete_old:bool=True)->None:
        """User-friendly gif-save function."""
        if self.log_screen:
            self.save_gif(f"{self.game_selector.selected_base_dir}run_t{int(time.time()):d}.gif",delete_old=delete_old,speedup=self.gif_speedup)
##################
### DEBUG TEXT ###
##################
    @lru_cache_func(maxsize=2048)
    def get_cached_map_name(self,map_id:int)->np.ndarray:
        """Return the map name."""
        placeholder=f"MAP_0x{int(map_id):03X}"
        return self.game_data.get("maps_names",{map_id:placeholder}).get(map_id,placeholder)
    def get_coordinates_text(self)->str:
        """Return full player coordinates text data."""
        return "M: 0x{:03X}\tY: {:03d}\tX: {:03d}\tD: {:01}({})\tG:[{:03d} {:03d}]\t I:[{:1d}, {:03d}, {:03d}] | {}".format(
            *[self.game_state["player_coordinates_data"][i] for i in range(4)],
            ["D","U","L","R"][self.game_state["player_coordinates_data"][3]],
            *[self.game_state["player_coordinates_data"][i] for i in range(9,11)],
            *[self.game_state["player_coordinates_data"][i] for i in range(4,7)],
            self.get_cached_map_name(self.game_state["player_coordinates_data"][0]
        ))
    def get_streamed_coordinates(self)->list[int]:
        """Return a list of coordinates for the stream_wrapper."""
        return [int(self.game_state["player_coordinates_data"][k]) for k in [2,1,0]]
    def get_powerup_button_text(self,btn_value:int,from_tk:bool):
        """Return the keyboard button combination used for the powerup."""
        num=btn_value%10
        off=max(0,btn_value-1)//10
        prefixes=["","CTRL-"] if from_tk else ["","^"]
        return f"{prefixes[off if off<len(prefixes) else -1]}{num:d}"
    def get_reward_text(self)->str:
        """Return the total_reward."""
        return f"{self.total_reward:.3f}"
    def get_game_debug_text(self)->str:
        """Game speficic debug text."""
        return "Game debug"
    def get_debug_text(self)->str:
        """Return all debug text."""
        return f"Map\t{self.get_coordinates_text()}\r\nReward\t{self.get_reward_text()}\r\n{self.get_game_debug_text()}\r\n{self.hook_get_debug_text()}"
    def get_game_commands_text(self)->str:
        """Game speficic buttons input text."""
        return ""
    def get_commands_text(self)->str:
        """Return button instructions for GUI play. Describes the action space."""
        if self.movement_max_actions==2:
            dir_txt="[1] Forward - [2] 90° Left + Forward"
        elif self.movement_max_actions==3:
            dir_txt="[1] Forward - [2] 90° Left + Forward - [3] 90° Right + Forward"
        else:
            dir_txt="[1] Down - [2] Left - [3] Right - [4] Up"
        extra_txt=""
        if self.action_nop and self.action_interact_id!=self.action_nop_id:
            extra_txt+=f"\tNo action:\t[{self.action_nop_id+1}]\n"
        if self.true_menu:
            extra_txt+=(f"\tInteractions:\t[{self.action_interact_id+1}] Interact - [{self.action_back_id+1}] Back"
                f"- [{self.action_menu_id+1}] Menu - [{self.action_extra_id+1}] Extra\n")
        elif self.button_interaction:
            extra_txt+=f"\tInteractions:\t[{self.action_interact_id+1}]\n"
            if self.bypass_powerup_actions:
                extra_txt+="!!!\tPowerup-actions are bypassed by interacting !!!\n"
            else:
                extra_txt+=f"\tPowerup-acts:\t[{', '.join([self.get_powerup_button_text(self.action_interact_id+2+i,False) for i,(_,_) in enumerate(self.event_rewards_powerups.items())])}]\n"
        txt=f"\n{'='*32}\n\tDirections:\t{dir_txt}\n{extra_txt}{self.get_game_commands_text()}\n\tActions internal indexing starts at (0), GUI buttons at [1].\n{'='*32}"
        return txt
####################
### DEBUG IMAGES ###
####################
    def change_screen_size(self,height:int,width:int)->None:
        """Changes the game screen. Used for the map drawer only, breaks during play."""
        self.centered_screen_size=np.array([height,width],dtype=np.int16,order="C")
        self.player_screen_position=(self.centered_screen_size-1)//2
        self.player_screen_bounds=np.vstack([-self.player_screen_position,self.centered_screen_size-self.player_screen_position],dtype=np.int16).transpose(1,0).flatten()
        self.game_state["game_minimal_screen"]=np.zeros(self.centered_screen_size.tolist(),dtype=np.uint8,order="C")
    def change_gui_render(self)->bool:
        """Changes the screen observation type for real-time GUI debugging."""
        if not self.from_gui or not self.gfx_loaded:
            return False
        can_use_hsv=self.env_config.get("hsv_image",False)
        if self.env_config.get("hsv_image",False):
            self.hsv_image=self.screen_observation_type==4 and not self.hsv_image
        self.screen_observation_type=4 if self.screen_observation_type==2 or (can_use_hsv and self.hsv_image) else 2
        if self.screen_observation_type<4:
            self.hsv_image=False
        self.env.use_gfx_image=self.screen_observation_type==4
        self.reset_gif_frames()
        return True
    def show_map_matrix(self,matrix:np.ndarray)->None:
        """Show a matrix in monocromatic mode."""
        show_image(map_matrix_to_image(matrix,self.tile_id_colors_lookup),"Map")
    def show_global_map(self)->None:
        """Show the legacy global map in monocromatic mode."""
        show_image(map_matrix_to_image(np.hstack(list(self.global_map),dtype=np.uint8),self.tile_id_colors_lookup),"Global map")
    def show_agent_screen(self)->None:
        """Show the current screen."""
        show_image(self.screen_ndarray(),title=self.get_coordinates_text())
    def show_agent_map(self)->None:
        """Show various stacked map data."""
        img=map_matrix_to_image(np.vstack([
            np.hstack(list(self.history_tracking["visited_maps"]),dtype=np.uint8),
            np.hstack([np.where(v.clip(0,1)>0,2,0).astype(np.uint8) for v in self.history_tracking["steps_on_map"]],dtype=np.uint8)
        ],dtype=np.uint8),self.tile_id_colors_lookup)
        show_image(img,title=self.get_coordinates_text())
    def show_last_gif_frames(self,frames:int=5)->None:
        """Show the last saved gif frames."""
        if len(self.gif_frames)>0:
            show_image(np.hstack(self.gif_frames[-frames:],dtype=np.uint8),f"Step: {self.step_count:d}")
        else:
            self.show_agent_screen()
##########################
### WARP DATA CHAINING ###
##########################
    def get_nearest_channel_warps(self,channel:int,map_id:int,y:int,x:int)->list:
        """Get a list of near warps given a channel."""
        warps=[]
        if map_id==channel:
            return [np.array([map_id,y,x],dtype=np.int16)]
        for w1 in self.get_cached_map_warps(map_id):
            if self.get_cached_global_map_channel(w1[2])==channel:
                warps.append(w1[-3:])
            else:
                for w2 in self.get_cached_map_warps(w1[2]):
                    if self.get_cached_global_map_channel(w2[2])==channel:
                        warps.append(w2[-3:])
                    else:
                        for w3 in self.get_cached_map_warps(w2[2]):
                            if self.get_cached_global_map_channel(w3[2])==channel:
                                warps.append(w3[-3:])
                            else:
                                continue
#                               for w4 in self.get_cached_map_warps(w3[2]):
#                                   if self.get_cached_global_map_channel(w4[2])==channel:warps.append(w4[-3:])
        return warps
    def get_nearest_channel_warp_from_player(self,channel:int,map_id:int,y:int,x:int)->list:
        """Return data relative to the nearest warp from the player at the given channel."""
        player_warps=self.get_nearest_channel_warps(channel,*self.game_state["player_coordinates_data"][:3])
        if len(player_warps)==0:
            player_coords=self.game_state["player_coordinates_data"][5:7].copy()
        else:
            player_coords=self.get_legacy_global_map_coords(*player_warps[0])### POSSIBLY IMPROVE SELECTION BY MAP_ID
        warps=self.get_nearest_channel_warps(channel,map_id,y,x)
        best_warp=[0,0,0]
        best_distance=900
        for w in warps:
            distances=player_coords-self.get_legacy_global_map_coords(*w)
            dist=np.abs(distances).sum()
            if dist<best_distance:
                (best_warp,best_distance)=(w,dist)
        return (best_warp,self.get_legacy_global_map_coords(*best_warp),best_distance)
#############################
### SCRIPTING INTERACTION ###
#############################
    def has_direct_script(self,func_name:str)->bool:
        """Checks if the scripting function name exists ."""
        return callable(get_function_from_self_or_scope(func_name,self,self.scripting_core_scope))
    def call_direct_script(self,func_name:str,*args,**kwargs)->bool:
        """Calls the scripting function. Return True to stopp execution of some scripts."""
        if func_name in self.called_step_func_names:
            return False
        func=get_function_from_self_or_scope(func_name,self,self.scripting_core_scope)
        ret=False
        if callable(func):
            ret=func(*args,**kwargs)
            if self.log_screen:
                self.add_gif_frame()
        self.called_step_func_names.add(func_name)
        return bool(ret)
    def set_automatic_scripted_map_ids(self,ids:list)->None:
        """Set map_id that calls a script once entered."""
        self.automatic_scripted_map_ids=np.array(ids,dtype=np.uint8)
    @lru_cache_func(maxsize=2048)
    def cached_is_automatic_scripted_map(self,map_id:int)->bool:
        """Return if the map has a startup script."""
        return map_id in self.automatic_scripted_map_ids
#        return nb_in_array_uint8(map_id,self.automatic_scripted_map_ids)
    def automatic_map_scripting(self,map_id:int)->bool:
        """Call the automatic scripting."""
        return self.call_direct_script("script_core_automatic_with_npc" if self.using_npcs else "script_core_automatic_without_npc",map_id)
##################
### CORE HOOKS ###
##################
    def core_hook_before_warp(self,global_warped:bool,movements:list)->None:
        """Executed before entering any warp."""
        self.npcs_map_tiles_update(self.game_state["player_coordinates_data"][0],True)
        if self.check_recursion(1):
            if self.has_agent():
                self.agent.hook_before_warp(global_warped,movements)
        self.call_direct_script("script_core_automatic_map_transition",movements[-1][0],global_warped=global_warped,teleported=False)
        self.hook_before_warp(global_warped,movements)
        self.game_state["previous_map_id"]=self.game_state["player_coordinates_data"][0]
    def core_hook_after_warp(self)->None:
        """Executed after exiting a warp."""
        self.npcs_map_tiles_update(self.game_state["player_coordinates_data"][0],True)
        if self.game_state["player_coordinates_data"][0] not in self.teleport_data and self.game_state["player_coordinates_data"][0] in self.game_data["teleport_data"]:
            self.teleport_data[self.game_state["player_coordinates_data"][0]]=self.game_data["maps"][self.game_state["player_coordinates_data"][0]][0]
        self.reset_forced_directional_movements()
        if self.has_agent():
            self.agent.reset_scheduled_actions()
        if self.check_recursion(1):
            if self.has_agent():
                self.agent.hook_after_warp()
        self.hook_after_warp()
    def check_new_event_reward_flags(self,check_button:bool=True)->bool:
        """Check if interactions awards new event flags. Mandatory for non NPC interaction."""
        if self.using_npcs or self.game_state["text_type"]>0 or self.game_state["battle_type"]>0:
            return False
        if check_button and self.button_interaction and self.last_action!=self.action_interact_id:
            return False
        for ek,ed in self.event_rewards_data_by_map.get(self.game_state["player_coordinates_data"][0],{}).items():
            if self.get_event_flag(ek)==0 and self.game_state["player_coordinates_data"][0]==ed[0][0] and len([1 for rk in ed[3] if self.get_event_flag(rk)!=1])==0 and np.array_equal(self.get_faced_direction_coordinates(*self.game_state["player_coordinates_data"][1:4]),self.get_faced_direction_coordinates(*ed[0][1:4])):
                ret=self.activate_event_reward_flag(ek,False)
                if ret:
                    self.update_overworld_screen()
                    return ret
        return False
    def check_script(self)->bool:
        """Check if a player is on a script tile and runs it."""
        for ek,ed in self.scripts_data_by_map.get(self.game_state["player_coordinates_data"][0],{}).items():
            if np.array_equal(self.game_state["player_coordinates_data"][:3],ed[0][:3]):
                func=get_function_from_self_or_scope(ed[1],self,self.scripting_core_scope)
                if not callable(func):
                    continue
                prev_map_data=self.game_state["player_coordinates_data"][:4].copy()
                self.check_new_event_reward_flags()
                should_delete_script=func()
                if self.log_screen and not np.array_equal(prev_map_data,self.game_state["player_coordinates_data"][:4]):
                    self.add_gif_frame()
                if should_delete_script:
                    deletable_keys=[]
                    for ekd,edd in self.scripts_data_by_map[ed[0][0]].items():
                        if edd[1]==ed[1]:
                            self.change_global_map_tile_from_local_coords(*edd[0][:3],old_script_tile_id,True)
                            deletable_keys.append(ekd)
                    for k in deletable_keys:
                        del self.scripts_data_by_map[ed[0][0]][k]
                self.hook_after_script(ek,ed,should_delete_script)
                return True
        return False
#########################
### TELEPORT-HANDLING ###
#########################
    def set_checkpoint_map_position(self,map_id:int,y:int=-1,x:int=-1,direction:int=4)->None:
        """Sets new checkpoint coordinates data."""
        if x<0 or y<0:
            self.last_checkpoint_place[:]=self.first_checkpoint_place if map_id==self.first_checkpoint_place[0] else [map_id,3,3,direction]
        else:
            self.last_checkpoint_place[:]=[map_id,y,x,direction]
    def force_teleport_to(self,map_id:int,y:int,x:int,direction:int=0,add_step:bool=True)->int:
        """Force the player to new coordinates."""
        coords=np.array([map_id,y,x,direction if direction<4 else self.game_state["player_coordinates_data"][3]],dtype=np.int16)
        if np.array_equal(self.game_state["player_coordinates_data"][:4],coords):
            return self.step_recursion_depth
        is_new_map=self.game_state["player_coordinates_data"][0]!=map_id
        if is_new_map:
            self.npcs_map_tiles_update(self.game_state["player_coordinates_data"][0],True)
###     self.step_recursion_depth+=1
        recursion_check=self.check_recursion(2)
        if self.game_state["player_coordinates_data"][0]!=map_id and recursion_check:
            self.core_hook_before_warp(True,[coords[:3]])
###     self.step_recursion_depth-=1
        self.game_state["player_coordinates_data"][:4]=coords
#        nb_assign_int16(self.game_state["player_coordinates_data"][:4],coords)
        self.game_state["player_coordinates_data"][4]=self.get_cached_global_map_channel(self.game_state["player_coordinates_data"][0])
        self.game_state["player_coordinates_data"][5:7]=self.get_cached_legacy_global_map_coords(self.game_state["player_coordinates_data"][0],self.game_state["player_coordinates_data"][1:3])
#        nb_assign_int16(self.game_state["player_coordinates_data"][5:7],self.get_cached_legacy_global_map_coords(self.game_state["player_coordinates_data"][0],self.game_state["player_coordinates_data"][1:3]))
        if self.game_state["player_coordinates_data"][4]==0:
            self.game_state["player_coordinates_data"][7:9]=self.game_state["player_coordinates_data"][5:7]
#            nb_assign_int16(self.game_state["player_coordinates_data"][7:9],self.game_state["player_coordinates_data"][5:7])
        self.game_state["player_coordinates_data"][9:11]=self.get_cached_global_map_coords(self.game_state["player_coordinates_data"][0],self.game_state["player_coordinates_data"][1:3])
#        nb_assign_int16(self.game_state["player_coordinates_data"][9:11],self.get_cached_global_map_coords(self.game_state["player_coordinates_data"][0],self.game_state["player_coordinates_data"][1:3]))
        if is_new_map:
            self.npcs_map_tiles_update(map_id,True)
        if not add_step:
            ret=self.step_recursion_depth
            self.update_overworld_screen()
            if recursion_check:
                self.hook_after_movement()
        elif recursion_check:
            ret=self.step_game_nop()
        if is_new_map:
            self.call_direct_script("script_core_automatic_map_transition",self.game_state["player_coordinates_data"][0],global_warped=False,teleported=True)
        return ret
    def teleport_to_event(self,event_name:str)->bool:
        """Teleport to given coordinates tied to an event."""
        event=self.event_rewards_data.get(event_name,[[]])[0]
        if len(event)<3:
            return False
        self.force_teleport_to(event[0],*self.get_faced_direction_coordinates(*event[1:3]))
        return True
    def teleport_to_checkpoint(self)->None:
        """Teleport to the last checkpoint coordinates."""
        self.force_teleport_to(*self.last_checkpoint_place[:3],4)#,0)
    def set_secondary_action_value(self,map_id:int)->None:
        """Assigns a secondary internal action value. Used for some powerups."""
        self.secondary_action_value=map_id if map_id in self.teleport_data else -1
    def powerup_teleport_to(self,map_id:int,force:bool=False)->None:
        """Legit teleport powerup."""
        if self.check_recursion(1) and (force or self.game_state["player_coordinates_data"][4]==0):
            self.secondary_action_value=-1
            if map_id in self.teleport_data:
                self.force_teleport_to(map_id,*self.teleport_data[map_id][0][::-1],0,add_step=False)
########################
### LOAD-SAVE STATES ###
########################
    def load_custom_save_from_events(self,starting_event:str,collected_flags:list)->bool:
        """Load a legit state starting from a custom event."""
        self.clear_events_caches()
        self.rewind_states_deque.clear()
        teleported=self.teleport_to_event(starting_event) if len(starting_event)>0 else False
        used_collected_flags=set()
        possible_levels={"new_game":0}
        checkpoint_map_data=self.first_checkpoint_place.copy()
        if teleported:
            checkpoint_map_data=self.event_rewards_data.get(starting_event,[[],self.first_checkpoint_place,[]])[1]
            def recursive_event_check(evt_name:str,depth:int,max_depth:int)->None:
                """Internal check to recursively extract data from chains of events."""
                if depth>=max_depth or len(used_collected_flags)>90:
                    return
                new_depth=min(depth+1,max_depth)
                for en in self.event_rewards_data.get(evt_name,[[],[],[],[]])[3]:
                    to_add=en not in used_collected_flags
                    if to_add:
                        used_collected_flags.add(en)
                        recursive_event_check(en,new_depth,max_depth)
            recursive_event_check(starting_event,0,len(self.event_rewards_data))
            used_collected_flags-=set([starting_event])
            self.call_direct_script("script_core_load_custom_save_from_starting_event",starting_event)
        possible_levels.update({evt_name:self.event_rewards_data.get(evt_name,[[],[],[0]])[2][0] for evt_name in list(used_collected_flags)+[starting_event]})
        used_level=np.max(list(possible_levels.values()))
        used_collected_flags=used_collected_flags.union(set(collected_flags))
        if teleported or len(used_collected_flags)>0:
            if len(used_collected_flags)>0:
                self.reset_event_flags()
            self.activate_event_reward_flag(self.first_event_name,True)
            self.game_state["player_level"]=used_level
            self.game_on_event_custom_load(starting_event,used_collected_flags,used_level)
            for k in used_collected_flags:
                self.activate_event_reward_flag(k,True)
        self.set_checkpoint_map_position(*checkpoint_map_data[:4])
        if self.using_npcs:
            self.update_npcs_movements_and_interactions(False)
        self.update_overworld_screen()
        return True
    def load_state(self,state:dict)->bool:
        """Load from a saved state dictionary."""
        if not isinstance(state,dict):
            return self.load_state(state.load_state()) if isinstance(state,GameCore) and state.check_recursion(1) else False
        reserved_names=set(self.get_reserved_attribute_states_names()) if self.true_menu and self.menu is not None else set()
        for k in self.get_attribute_state_names():
            if k in state and len(k)>0 and (hasattr(self,k) or k in reserved_names):
                if k=="reserved_menu_state":
                    if self.true_menu and self.menu is not None and isinstance(state[k],dict):
                        for sn,sv in state[k].items():
                            if len(sn)>0 and hasattr(self.menu,sn):
                                setattr(self.menu,sn,deepcopy(sv))
                else:
                    setattr(self,k,deepcopy(state[k]))
        self.clear_events_caches()
        self.text_changed_during_step=True
        self.menu_changed_during_step=True
        return True
    def rewind_state(self,saved_steps:int=1)->bool:
        """Load from internal save states of previous steps."""
        if len(self.rewind_states_deque)>1:
            try:
                limit=len(self.rewind_states_deque)-1
                for _ in range(min(max(1,saved_steps),limit)-1):
                    self.rewind_states_deque.pop()
                return self.load_state(self.rewind_states_deque[len(self.rewind_states_deque)-1])
            except IndexError:
                pass
        return False
    def save_state(self)->dict:
        """Save the current state to a dictionary."""
        state={k:deepcopy(getattr(self,k)) for k in self.get_attribute_state_names() if len(k)>0 and hasattr(self,k)}
        if self.true_menu and self.menu is not None:
            state["reserved_menu_state"]={k:deepcopy(getattr(self.menu,k)) for k in self.menu.get_attribute_state_names() if len(k)>0 and hasattr(self.menu,k)}
        return state
    def load_state_from_file(self,file_path:str)->dict:
        """Load a state from a pickle dumped file."""
        data=pickle_load(file_path)
        if data is None:
            data={}
        ret=False if len(data)<len(self.get_attribute_state_names()) else self.load_state(data)
        if ret:
            self.load_custom_save_from_events("",self.get_active_flags_names())
        return ret
    def save_state_to_file(self,file_path:str)->bool:
        """Save a state into a pickle dump file."""
        return pickle_save(file_path,self.save_state())
####################
### ACTION SPACE ###
####################
    def get_current_max_action_space(self)->int:
        """Returns the max value of the action space depending on powerups."""
        return self.action_interact_id
    def roll_random_screen_offsets(self,count:int=1)->np.ndarray:
        """Return a random [y,x] offset from player inside the screen view."""
        return np.hstack([np.random.randint(self.player_screen_bounds[0],self.player_screen_bounds[1]+1,(count,1)),np.random.randint(self.player_screen_bounds[2],self.player_screen_bounds[3]+1,(count,1))])
    def roll_random_directions(self,count:int=1)->int:
        """Return a random direction action."""
        return np.random.randint(0,self.movement_max_actions+1,count)
    def roll_random_actions(self,count:int=1)->int:
        """Return a random action depending on obtained powerups."""
        return np.random.randint(0,self.get_current_max_action_space()+1,count)
    def roll_random_actions_without_nop(self,count:int=1)->int:
        """Return a random action avoiding doing nothing."""
        if not self.action_nop:
            return self.roll_random_actions(count)
        actions=np.random.randint(0,self.get_current_max_action_space(),count)
        actions[actions>self.action_nop_id]+=1
        return actions
    def predict_action(self)->int:
        """Predict an action if there is an agent linked or pick one randomly otherwise."""
        return self.roll_random_actions_without_nop(1)[0] if self.agent is None else self.agent.predict_action()
##################
### DATA NAMES ###
##################
    def get_game_attribute_state_names(self)->list[str]:
        """List of game-specific attribute names preserved in a save state."""
        return []
    def get_extra_attribute_state_names(self)->list[str]:
        """List of extra attribute names preserved in a save state."""
        return []
    def get_reserved_attribute_states_names(self)->list[str]:
        """List of reserved attributes names for child objects preserved in a save state."""
        return ["reserved_menu_state"]#,"reserved_agent_state"]
    def get_attribute_state_names(self)->set:
        """List of all attribute names preserved in a save state."""
        return set(["seed","step_count","game_completed","total_reward",
            "secondary_action_value","forced_directional_movements","teleport_data",
            "global_map_shapes","global_map_padding","global_map_gfx_padding","global_map",
            "event_rewards_powerups","event_rewards_data","scripts_data","npcs_data",
            "event_flags","event_rewards_data_by_map","scripts_data_by_map","npcs_data_by_map",
            "npcs_map_lookup","last_npc_name","event_flags_lookup","last_checkpoint_place",
            "history_tracking","game_state","stacked_state",
            "text_queue","text_changed_during_step","menu_changed_during_step",
            "menu_current_depth","menu_content_by_depth","menu_bg_by_depth","menu_cursor_data",
        ]+self.get_reserved_attribute_states_names()+
        list(self.get_game_attribute_state_names())+list(self.get_extra_attribute_state_names()))
##################################
### ENVIRONMENT-LIKE UTILITIES ###
##################################
    def env_is_done(self)->bool:
        """Extra conditions to the game execution."""
        return False
    def env_reward(self,action:int=-1)->float:
        """Total reward of the action at the current step."""
        return 5.*self.get_event_flags_sum()+25.*self.game_completed
    def update_reward(self,action:int=-1)->float:
        """Delta reward of the action at the current step."""
        self.check_new_event_reward_flags()
        new_total=self.env_reward(action)
        new_step=new_total-self.total_reward
        self.total_reward=new_total
        return new_step
    def is_done(self)->bool:
        """Return True to stop the game execution."""
        return (self.max_steps>0 and self.step_count>=self.max_steps) or (self.game_completed and not self.infinite_game) or self.env_is_done()
    def close(self)->None:
        """Cleanup code for freeing environment resources."""
        return
    def get_screen_observation(self)->np.ndarray:
        """Return the screen ndarray according to the screen_observation_type config."""
        if self.screen_observation_type==0:
            img=np.zeros((1,1),dtype=np.uint8,order="C")
        elif self.screen_observation_type==1:
            img=self.game_state["game_minimal_screen"]
        elif self.screen_observation_type==2:
            img=to_categorical(self.game_state["game_minimal_screen"],count_tiles_ids,np.uint8)
        elif self.screen_observation_type in (3,4):
            img=self.get_agent_screen()
        else:
            img=self.game_state["game_minimal_screen"]
        return np.array(img,dtype=np.uint8)
    def get_nonscreen_observations(self)->dict[np.ndarray]:
        """Main method to declare the step observations."""
        return {}
    def get_nonscreen_observation_space(self)->dict:
        """Declarations of the get_nonscreen_observations space types."""
        return {}
    def get_observations(self)->dict[np.ndarray]:
        """Return the observation dictionary."""
        obs=self.get_nonscreen_observations()
        if self.auto_screen_obs:
            obs["screen"]=self.get_screen_observation()
        return obs
##################
### RESET GAME ###
##################
    def game_on_reset(self)->None:
        """Game-specific reset to initial state."""
        return
    def reset_game(self,seed:Union[int,None]=None)->None:
        """Reset the game to the initial state."""
        self.set_game_seed(seed)
        (map_id,local_coords)=(self.start_map_position[0],self.start_map_position[1:3].astype(np.int16))
        self.last_action=0
        self.game_state["pseudo_seed"]=np.random.randint(0,0x100,dtype=np.int16)
        self.game_state["player_coordinates_data"][:]=np.asarray([map_id]+local_coords.tolist()+[self.start_map_position[3]]+[self.get_cached_global_map_channel(map_id)]+self.get_cached_legacy_global_map_coords(map_id,local_coords).tolist()+self.get_nearest_channel_warp_from_player(0,0,6,5)[1].tolist()+self.get_cached_global_map_coords(map_id,local_coords).tolist(),dtype=np.int16)
        self.game_state["last_suggested_coordinates"][:]=0
        self.game_state["previous_map_id"]=map_id
        self.reset_forced_directional_movements()
        self.teleport_data.clear()
        if self.has_menu():
            self.menu.reset(seed)
        self.event_rewards_powerups=deepcopy(self.game_data["powerups"])
        self.event_rewards_data=deepcopy(self.filter_dict_cache(self.game_data["events"]))
        self.scripts_data=deepcopy(self.filter_dict_cache(self.game_data["scripts"]))
        self.npcs_data=deepcopy(self.game_data["npcs"]) if self.using_npcs else {}
        self.npcs_data_by_map.clear()
        for k,v in self.npcs_data.items():
            self.npcs_data_by_map[v[0][0]]={}
        for k,v in self.npcs_data.items():
            self.npcs_data_by_map[v[0][0]][k]=[
                np.array(v[0],dtype=np.int16),np.repeat(np.expand_dims(np.array(v[0][1:4],dtype=np.int16),axis=0),2,axis=0)
            ]+[v[1]]+[-1,0,0,0]+v[2:]
        self.event_rewards_data_by_map.clear()
        for k,v in self.event_rewards_data.items():
            self.event_rewards_data_by_map[v[0][0]]={}
        for k,v in self.event_rewards_data.items():
            self.event_rewards_data_by_map[v[0][0]][k]=[np.array(v[0],dtype=np.int16)]+v[1:]+[[nwd for c in range(2) for nwd in self.get_nearest_channel_warp_from_player(c,*v[0][:3])[:2]]]
        self.scripts_data_by_map.clear()
        for k,v in self.scripts_data.items():
            self.scripts_data_by_map[v[0][0]]={}
        for k,v in self.scripts_data.items():
            self.scripts_data_by_map[v[0][0]][k]=[np.array(v[0],dtype=np.int16)]+v[1:]
        self.last_npc_name=""
        self.history_tracking["visited_maps"][:]=0
        self.history_tracking["steps_on_map"][:]=0
        self.game_state["game_minimal_screen"][:]=0
        self.game_state["powerup_walk_tile"]=0

        for k in ["powerup_walk_tile","powerup_screen_remove_tile","powerup_started",
            "player_level","text_type","battle_type","loss_count"
        ]:
            self.game_state[k]=0
        for _,v in self.stacked_state.items():
            v[0,:]=0
        self.game_on_reset()

        self.reset_event_flags()
        ### self.load_custom_save_from_events WAS INITIALLY HERE, CAUSING A BUG WITH NPC NOT BEING INITIALLY VISIBLE, BUT INTERACTABLE
        self.global_map[:]=no_map_tile_id
        self.build_global_map(False)
        if not self.load_custom_save_from_events(self.starting_event,self.starting_collected_flags):
            self.update_overworld_screen()
        self.rewind_states_deque.clear()
        self.reset_gif_frames()
        self.hook_reset()
        for _,v in self.stacked_state.items():
            v[1:,:]=v[0]
        self.step_count=0
        self.game_completed=False
        if hasattr(self,"game_state") and hasattr(self,"env_config"):
            self.define_game_config_post_game_state_creation(self.env_config)
        self.update_reward()
        if self.has_agent():
            self.agent.reset()
        self.call_direct_script("script_core_reset")
        self.clear_all_lru_cache()
##################
### GAME LOGIC ###
##################
    def handle_menu_actions(self,action:int)->None:
        """Perform a step in the menu."""
        if self.has_menu():
            self.menu.step_menu(action)
            #self.menu_changed_during_step=True
    def handle_non_directional_actions(self,action:int)->tuple:
        """Game-specific logic for non-movement buttons."""
        return (True,False)
    def game_handle_random_encounter_spawn(self,map_id:int,tile:int)->int:
        """Handles random encounters. Return 0 without encounters, 1 on win, -1 on loss"""
        return 0
    def game_bypass_powerup_tiles(self,tile:int)->None:
        """Automatic bypass actions of powerups if event prerequisites are met."""
        return
    def game_powerup_fix_tile(self,tile:int,powerup_started:bool)->int:
        """Fix the tile for powerup-logic purposes."""
        return tile
    def game_powerup_first_time_use_events(self,action:int,tile:int)->None:
        """Set events for first-time powerup usage."""
        return
    def game_post_movement_powerup_status(self,tile:int)->None:
        """Finalizes the state of powerup-usage."""
        return
    def game_after_step(self,action:int)->None:
        """Runs after the a game step."""
        return
    def run_action_on_emulator(self,action:int=-1)->bool:
        """Process one game frame."""
        self.text_changed_during_step=False
        self.menu_changed_during_step=False
        if self.game_state["menu_type"]>0:
            ret=self.run_menu_action(action)
        elif self.game_state["text_type"]==0:
            if self.has_menu() and self.menu.force_menu:
                ret=self.run_menu_action(action)
            else:
                ret=self.run_overworld_action(action)
        elif self.game_state["battle_type"]>0:
            if not self.true_menu and (not self.button_interaction or self.benchmarking):
                self.game_state["battle_type"]=0
            ret=self.run_battle_action(action)
        else:
            if not self.true_menu and (not self.button_interaction or self.benchmarking):
                self.game_state["text_type"]=0
                ret=self.run_overworld_action(action)
            else:
                ret=self.run_text_action(action)
                self.update_text_screen()
        if self.log_screen:
            self.add_gif_frame()
        if ret and self.game_state["battle_type"]==0 and active_script_tile_id==self.global_map[self.game_state["player_coordinates_data"][4],self.game_state["player_coordinates_data"][5],self.game_state["player_coordinates_data"][6]]:
            self.check_script()
        self.game_after_step(action)
        self.hook_after_step(action)
        return ret
    def run_menu_action(self,action:int=-1)->bool:
        """Process any menu-related action."""
        up_overworld=False
        prev_menu=0x40*self.game_state["menu_type"]+self.game_state["sub_menu_type"]
        if not self.true_menu:
            self.game_state["menu_type"]=0
            self.game_state["sub_menu_type"]=0
            self.clear_menu_content(0)
        elif self.game_state["menu_type"]==0:
            if action==self.action_menu_id:
                self.clear_menu_content(0)
                self.handle_menu_actions(action)
                if self.game_state["menu_type"]==0:
                    self.game_state["menu_type"]=2
            elif self.has_menu() and self.menu.force_menu:
                self.handle_menu_actions(self.action_menu_id)
        elif self.game_state["menu_type"]==2:
            if action in {self.action_back_id,self.action_menu_id}:
                self.clear_menu_content(0)
                self.handle_menu_actions(self.action_back_id)
                if self.game_state["menu_type"]!=0:
                    self.game_state["menu_type"]=0
                    self.game_state["sub_menu_type"]=0
                up_overworld=True
            else:
                self.handle_menu_actions(action)
                up_overworld=(self.screen_observation_type<4 and self.game_state["menu_type"]==0)
        else:
            self.handle_menu_actions(action)
            up_overworld=(self.screen_observation_type<4 and self.game_state["menu_type"]==0)
        if up_overworld:
            self.update_overworld_screen()
            if self.screen_observation_type<4:
                if self.game_state["text_type"]>0:
                    self.update_text_screen()
        elif self.game_state["menu_type"]>0:
            if prev_menu!=self.game_state["menu_type"]+100*self.game_state["sub_menu_type"]:
                self.update_overworld_screen()
            self.update_menu_screen()
        return False
    def run_overworld_action(self,action:int=-1)->bool:
        """Process any overworld-related action."""
        (forced_direction,skip_movement,powerup_started,should_check_script,warped,screen_update_fallback,npc_script_name,npc_script_args)=(
            len(self.forced_directional_movements)>0,False,False,True,False,True,None,[])
        if forced_direction:
            action=self.forced_directional_movements.pop()
            if action>=4:
                forced_direction=False
        if not forced_direction:
            if action<0:
                action=self.action_nop_id
            exp_dir_mov=self.get_expected_direction_and_movement(self.movement_max_actions,action,
                self.game_state["player_coordinates_data"][3])
            if action>=self.movement_max_actions:
                if self.true_menu and action==self.action_menu_id and self.game_state["menu_type"]==0:
                    return self.run_menu_action(self.action_menu_id)
                (skip_movement,powerup_started)=self.handle_non_directional_actions(action)
            else:
                self.game_state["player_coordinates_data"][3]=exp_dir_mov[2]
        else:
            exp_dir_mov=self.get_expected_direction_and_movement(4,
                self.get_action_from_direction_offsets_4way(*self.get_direction_offset(action
                )),self.game_state["player_coordinates_data"][3])
            self.game_state["player_coordinates_data"][3]=exp_dir_mov[2]
            should_check_script=False
        if not skip_movement:
            new_expected_coords=self.game_state["player_coordinates_data"][:3].copy()
            new_expected_global_coords=self.game_state["player_coordinates_data"][4:7].copy()
            new_expected_coords[1:3]+=exp_dir_mov[:2]
#            nb_sum_int16(new_expected_coords[1:3],exp_dir_mov[:2])
            new_expected_global_coords[1:3]+=exp_dir_mov[:2]
#            nb_sum_int16(new_expected_global_coords[1:3],exp_dir_mov[:2])
            (global_warped,expected_tile,map_dims)=(False,0,self.get_cached_map_sizes(self.game_state["player_coordinates_data"][0]))
            if self.cached_is_point_inside_area(*new_expected_coords[1:3],0,0,*map_dims):
                expected_tile=self.global_map[new_expected_global_coords[0],new_expected_global_coords[1],new_expected_global_coords[2]]
            elif self.game_state["player_coordinates_data"][4]==0:
                if self.cached_is_point_inside_area(*new_expected_global_coords[1:3],0,0,*self.global_map.shape[1:3]):
                    temp_expected_tile=self.global_map[new_expected_global_coords[0],new_expected_global_coords[1],new_expected_global_coords[2]]
                    if self.is_walkable_tile_id(temp_expected_tile,self.game_state["powerup_walk_tile"]):
                        new_expected_coords[0]=self.cached_check_point_bounds_map(new_expected_coords[0],*new_expected_global_coords[1:3])
                        if new_expected_coords[0]!=self.game_state["player_coordinates_data"][0]:
                            global_warped=True
                            expected_tile=temp_expected_tile
                            new_expected_coords[1:3]=new_expected_global_coords[1:3]-self.get_cached_global_map_bounds(new_expected_coords[0])[:2]
            if self.bypass_powerup_actions and action<self.movement_max_actions:
                self.game_bypass_powerup_tiles(expected_tile)
            expected_tile=self.game_powerup_fix_tile(expected_tile,powerup_started)
            if self.is_walkable_tile_id(expected_tile,self.game_state["powerup_walk_tile"]) and self.special_tile_direction_check(expected_tile,self.game_state["player_coordinates_data"][3]):
                self.game_powerup_first_time_use_events(action,expected_tile)
                movements=[new_expected_coords]
                if expected_tile==warp_tile_id and self.global_map[self.game_state["player_coordinates_data"][4],self.game_state["player_coordinates_data"][5],self.game_state["player_coordinates_data"][6]]!=warp_tile_id:
                    for w in self.get_cached_map_warps(self.game_state["player_coordinates_data"][0]):
                        if np.array_equal(new_expected_coords[1:3],w[:2]):
                            movements.append(w[-3:])
                            break
                if self.game_handle_random_encounter_spawn(movements[-1][0],expected_tile)<0:
                    warped=True
                else:
                    self.game_post_movement_powerup_status(expected_tile)
                    screen_update_fallback=False
                    warped=global_warped or len(movements)>1
                    if warped:
                        self.core_hook_before_warp(global_warped,movements)
                    for m in movements:
                        self.game_state["player_coordinates_data"][:3]=m[:3]
#                        nb_assign_int16(self.game_state["player_coordinates_data"][:3],m[:3])
                        self.game_state["player_coordinates_data"][4]=self.get_cached_global_map_channel(self.game_state["player_coordinates_data"][0])
                        self.game_state["player_coordinates_data"][5:7]=self.get_cached_legacy_global_map_coords(self.game_state["player_coordinates_data"][0],self.game_state["player_coordinates_data"][1:3])
#                        nb_assign_int16(self.game_state["player_coordinates_data"][5:7],self.get_cached_legacy_global_map_coords(self.game_state["player_coordinates_data"][0],self.game_state["player_coordinates_data"][1:3]))
                        if self.game_state["player_coordinates_data"][4]==0:
                            self.game_state["player_coordinates_data"][7:9]=self.game_state["player_coordinates_data"][5:7]
#                            nb_assign_int16(self.game_state["player_coordinates_data"][7:9],self.game_state["player_coordinates_data"][5:7])
                        self.game_state["player_coordinates_data"][9:11]=self.get_cached_global_map_coords(self.game_state["player_coordinates_data"][0],self.game_state["player_coordinates_data"][1:3])
#                        nb_assign_int16(self.game_state["player_coordinates_data"][9:11],self.get_cached_global_map_coords(self.game_state["player_coordinates_data"][0],self.game_state["player_coordinates_data"][1:3]))
                        self.history_tracking["steps_on_map"][self.game_state["player_coordinates_data"][4],self.game_state["player_coordinates_data"][5],self.game_state["player_coordinates_data"][6]]+=1
                    self.update_overworld_screen()
                    self.hook_after_movement()
        if not warped:
            if self.using_npcs:
                (npc_script_name,npc_script_args)=self.update_npcs_movements_and_interactions()
                self.npcs_map_tiles_update(self.game_state["player_coordinates_data"][0],warped)
        else:
            self.core_hook_after_warp()
        if screen_update_fallback:
            self.update_overworld_screen()
            self.hook_after_movement()
        if not skip_movement and self.cached_is_automatic_scripted_map(self.game_state["player_coordinates_data"][0]):
            self.automatic_map_scripting(self.game_state["player_coordinates_data"][0])
        if npc_script_name is not None:
            should_check_script=False
            self.call_direct_script(npc_script_name,*npc_script_args)
            if not npc_script_name.startswith("script_npc_text"):
                self.call_direct_script("script_npc_text")
        if self.game_state["menu_type"]>0:
            self.update_menu_screen()
        elif self.game_state["text_type"]>0:
            self.update_text_screen()
        return should_check_script
    def run_battle_action(self,action:int=-1)->bool:
        """Process any battle-related action."""
        if action==self.action_interact_id:
            self.game_state["battle_type"]=0
            self.game_state["text_type"]=1
        return False
    def run_text_action(self,action:int=-1)->bool:
        """Process any text-related action."""
        if action==self.action_interact_id:
            if self.has_menu():
                self.menu.return_to_overworld()
            self.game_state["text_type"]=0
            self.update_overworld_screen()
        else:
            self.update_text_screen()
        return False
###########################
### STEP GAME FUNCTIONS ###
###########################
    def step_game(self,action:int=-1)->int:
        """Step the game one frame. Safer function wrapper."""
        self.step_recursion_depth+=1
        rec_check=self.check_recursion(1)
        if rec_check:
            self.last_action=action
        self.run_action_on_emulator(action)
        if rec_check:
            self.called_step_func_names.clear()
            self.step_count+=1
            if self.step_count%65536==0:
                self.clear_all_lru_cache()
        self.step_recursion_depth-=1
        if rec_check and (len(self.rewind_states_deque)==0 or (self.rewind_steps>0 and self.step_count%self.rewind_steps==0)):
            self.rewind_states_deque.append(self.save_state())
        return self.step_recursion_depth
    def step_game_predict(self)->int:
        """Step the game following the agent prediction."""
        return self.step_game(self.predict_action())
    def step_game_in_direction(self,direction:int)->int:
        """Step the game moving in the given direction."""
        coords=self.game_state["player_coordinates_data"][:3].copy()
        coords[1:3]+=self.get_direction_offset(direction)
        return self.force_teleport_to(*coords,direction)
    def step_game_down(self)->int:
        """Step the game moving down."""
        return self.step_game_in_direction(0)
    def step_game_left(self)->int:
        """Step the game moving left."""
        return self.step_game_in_direction(2)
    def step_game_right(self)->int:
        """Step the game moving righ."""
        return self.step_game_in_direction(3)
    def step_game_up(self)->int:
        """Step the game moving up."""
        return self.step_game_in_direction(1)
    def step_game_invert_direction(self)->int:
        """Step the game moving in the opposite of the current direction."""
        return self.step_game_in_direction(self.get_inverted_direction(self.game_state["player_coordinates_data"][3]))
    def step_game_invert_direction_keep_facing(self)->int:
        """Step the game moving in the opposite of the current direction, preserving facing."""
        prev_direction=self.game_state["player_coordinates_data"][3]
        ret=self.step_game_in_direction(self.get_inverted_direction(self.game_state["player_coordinates_data"][3]))
        self.game_state["player_coordinates_data"][3]=prev_direction
        return ret
    def step_game_nop(self)->int:
        """Step the game without doing anything."""
        return self.step_game(self.action_nop_id if self.action_nop and self.action_interact_id!=self.action_nop_id else -1)
    def step_game_interact(self)->int:
        """Step the game pressing selecting the interaction action."""
        return self.step_game(self.action_interact_id)

if __name__=="__main__":
    from cli import main_cli
    test_game_name=1
    test_mode="play"
    test_action_complexity=4
    test_screen_observation_type=4
    test_screen_mult=2
    test_starting_event="medal1"
    test_stdout=True
    main_cli([test_game_name,"--mode",test_mode,
        "--action_complexity",test_action_complexity,
        "--screen_observation_type",test_screen_observation_type,
        "--screen_view_mult",test_screen_mult,
        "--starting_event",test_starting_event,
        ]+(["-stdout"] if test_stdout else []))
