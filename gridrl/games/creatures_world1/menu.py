#!/usr/bin/env python3

"""Menu implementation of the game [creatures_world1]."""

from typing import Union
from enum import IntEnum
import sys
import os
#import cython
import numpy as np

if __package__ is None or len(__package__)==0:
    try:
        __file__=__file__
    except NameError:
        __file__=""
    sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}{os.sep}..{os.sep}..")
try:
    from gridrl.core_environment import GridRLAbstractEnv
    from gridrl.menu_base import M_actions,M_overworld,M_unknown,M_text,M_menu
    from gridrl.games.creatures_world1.constants import (
        moves_list,creatures_names,items_list,
    )
except ModuleNotFoundError:
    from core_environment import GridRLAbstractEnv
    from menu_base import MenuBase,M_actions,M_overworld,M_unknown,M_text,M_menu
    from games.creatures_world1.constants import (
        moves_list,creatures_names,items_list,
    )

sys.dont_write_bytecode=True

if True:#not cython.compiled:
    M_main_menu=IntEnum("M_main_menu",["m_overworld","m_unknown","m_text","m_menu","m_tracker","m_party",
        "m_bag","m_player","m_save","m_settings","m_battle","m_pc"],start=0)
    M_tracker=IntEnum("M_tracker",["tracker","side_menu","data","area"],start=0)
    M_party=IntEnum("M_party",["party","menu_selection","stats","switch_selection","cant_field_move","move_teaching"],start=0)
    M_bag=IntEnum("M_bag",["bag","use_toss","cant_toss","toss_count","toss_yesno","tossed","cant_use",
        "use_overworld","use_item","use_creature_selection","used_on_creature","cant_on_creature",
        "lv_up_item","evolution_item","move_selection","used_on_move",
        "move_yesno","move_creature_selection","cant_learn_move","duplicate_move","teached_move_new",
        "teaching_move_yesno","teaching_quit_yesno","teaching_move_quit","teaching_move","cant_delete_move","teached_move_replaced",
    ],start=0)
    M_player=IntEnum("M_player",["player"],start=0)
    M_save=IntEnum("M_save",["save"],start=0)
    M_settings=IntEnum("M_settings",["settings"],start=0)
    M_battle=IntEnum("M_battle",["battle"],start=0)
    M_pc=IntEnum("M_pc",["pc"],start=0)

class CreaturesWorld1Menu(MenuBase):
    """Menu class of the game creatures_world1."""
    def __init__(self,env:GridRLAbstractEnv=None,*args,**kwargs)->None:
        """Constructor."""
        super().__init__(env,*args,**kwargs)
        self.repr_menu_enum={(k.value+256)%256:k.name for k in M_main_menu}
        self.repr_submenu_enum={(em.value+256)%256:{(k.value+256)%256:k.name for k in es} for em,es in zip(M_main_menu,[M_overworld,M_unknown,M_text,M_menu,M_tracker,M_party,M_bag,M_player,M_save,M_settings,M_battle,M_pc])}
        self.actions={"down":0,"left":1,"right":2,"up":3,"a":4,"b":5,"start":6,"select":7,
            "d":0,"l":1,"r":2,"u":3,"s":6,"z":7,
            "0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,
        }
        self.header_printed=False
        self.menu_state_cursor_creature=np.uint8(0)
        self.menu_state_cursor_bag=np.uint8(0)
        self.menu_state_cursor_roll_bag=np.uint8(0)
        self.reset()
    def get_extra_attribute_state_names(self)->list[str]:
        """List of extra attribute names preserved in a save state."""
        return ["menu_state_cursor_creature","menu_state_cursor_bag","menu_state_cursor_roll_bag"]
    def str_to_action(self,name:Union[str,int])->int:
        """Conversion from readable inputs to action space."""
        return self.actions.get(name,-1) if isinstance(name,str) else int(name)
    def get_observations(self)->np.ndarray:
        """Menu step cursor observations."""
        return np.array([
            self.menu_state_menu,
            self.menu_state_sub_menu,
            self.menu_state_main_menu_cursor,
            self.menu_state_cursor_x,
            self.menu_state_cursor_y,
            self.menu_state_cursor_roll_y,
            self.menu_state_cursor_roll_max,
            self.menu_state_cursor_yesno,
            self.menu_state_pending_text_presses_count,
            self.menu_state_cursor_creature,
            self.menu_state_cursor_bag,
            self.menu_state_cursor_roll_bag,
        ],dtype=np.uint8)
    def save_cursor_creature(self,cursor_name:str)->None:
        """Update the creature cursor value."""
        if cursor_name=="cursor_x":
            self.menu_state_cursor_creature=self.menu_state_cursor_x
        else:
            self.menu_state_cursor_creature=self.menu_state_cursor_y
    def save_cursor_bag(self)->None:
        """Update the bag cursor values."""
        self.menu_state_cursor_roll_bag=self.menu_state_cursor_roll_y
        self.menu_state_cursor_bag=self.menu_state_cursor_y
    def get_tracker_text(self):
        """Text for trackers menu names list."""
        text_list=["Tracked Creatures"]
        for i in range(self.menu_state_cursor_roll_y+1,min(self.env.game_state["tracker_flags"].shape[0],
            self.menu_state_cursor_roll_y+self.menu_state_cursor_roll_max+1
        )):
            text_list.append(f"{i:03d} {creatures_names[i] if self.env.game_state['tracker_flags'][i]>0 else '--------'}")
        return text_list
    def get_party_text(self):
        """Text for party menu names list."""
        text_list=[f"{creatures_names[idx][:10]:10.10} {np.ceil(100.*self.env.game_state['party_hp_ratios'][pos]):03.0f} {self.env.game_state['party_levels'][pos]:03d}" for pos,idx in enumerate(self.env.game_state["party_index"][:self.env.game_state["party_size"]])]
        if len(text_list)==0:
            text_list.append("Party is empty.")
        return text_list
    def get_party_moves_text(self,idx:int):
        """Text of moves on the selected creature."""
        text_list=[moves_list[i].title() for i in self.env.game_state["party_moves"][idx]]
        return text_list
    def get_party_stats_text(self,idx:int):
        """Text of stats on the selected creature."""
        text_list=self.get_party_moves_text(idx)
        return text_list
    def get_bag_text(self):
        """Text for bag menu names list."""
        text_list=[f"{items_list[i]} x {q:d}" for i,q in self.env.game_state["bag"][self.menu_state_cursor_roll_y:min(
            self.env.game_state["bag_size"],self.menu_state_cursor_roll_y+self.menu_state_cursor_roll_max+1
        )]]
        if len(text_list)<self.menu_state_cursor_roll_max+1:
            text_list.append("Cancel")
        return text_list
    def get_player_text(self):
        """Text for player menu."""
        return ["Player",f"Money: {self.env.game_state['money']:d}"]#,f"Steps: {self.env.step_count:d}"]
    def return_to_bag_menu(self)->None:
        self.set_menu(M_main_menu.m_bag,M_bag.bag)
        self.env.set_new_menu_layer(1,2,-2,self.env.centered_screen_size[1],
            y_cursor=2,x_cursor=2,vertical=True,value=self.menu_state_cursor_bag,clear_until_depth=1)
        self.env.append_multiple_menu_contents(2,3,True,self.get_bag_text())
    def no_menu_overworld(self,act:int)->None:
        """Logic without menu active."""
        if act==M_actions.start or self.force_menu:
            self.return_to_main_menu(0)
            self.env.set_new_menu_layer(0,-5,*self.env.centered_screen_size.tolist(),
                y_cursor=1,x_cursor=-5,vertical=True,value=self.menu_state_main_menu_cursor,clear_until_depth=0)
            self.env.append_multiple_menu_contents(1,-4,True,["Tracker","Party","Bag","Player","Save","Settings","Exit"])
    def menu_start(self,act:int)->None:
        """Logic of main menu."""
        if act in {M_actions.b,M_actions.start}:
            self.return_to_overworld()
        elif act==M_actions.down:
            self.move_cursor_infinite("main_menu_cursor",1,7)
        elif act==M_actions.up:
            self.move_cursor_infinite("main_menu_cursor",-1,7)
        elif act==M_actions.a:
            if self.menu_state_main_menu_cursor>=6:
                self.return_to_overworld()
            else:
                self.set_menu(self.menu_state_main_menu_cursor+4,0)
                self.menu_state_pending_text_presses_count=0
                if self.menu_state_menu==4:#M_main_menu.m_tracker:
                    if self.env.get_event_flag("encounters_tracker")>0:
                        self.menu_state_cursor_roll_max=7
                        self.menu_state_cursor_roll_y=0
                        self.menu_state_cursor_y=0
                        self.env.set_new_menu_layer(0,0,*self.env.centered_screen_size.tolist(),
                            y_cursor=1,x_cursor=0,vertical=True,value=0,clear_until_depth=1)
                        self.env.append_multiple_menu_contents(0,1,True,self.get_tracker_text())
                    else:
                        self.return_to_overworld()
                elif self.menu_state_menu==5:#M_main_menu.m_party:
                    if self.env.get_event_flag("start_decision")>0:
                        self.menu_state_cursor_y=self.menu_state_cursor_creature
                        self.env.set_new_menu_layer(0,0,*self.env.centered_screen_size.tolist(),
                            y_cursor=0,x_cursor=0,vertical=True,value=self.menu_state_cursor_creature,clear_until_depth=1)
                        self.env.append_multiple_menu_contents(0,1,True,self.get_party_text())
                    else:
                        self.return_to_overworld()
                elif self.menu_state_menu==6:#M_main_menu.m_bag:
                    self.menu_state_cursor_roll_max=3
                    self.menu_state_cursor_roll_y=self.menu_state_cursor_roll_bag
                    self.menu_state_cursor_y=self.menu_state_cursor_bag
                    self.return_to_bag_menu()
                else:
                    self.env.set_new_menu_layer(0,0,*self.env.centered_screen_size.tolist(),displayed_cursor=False,clear_until_depth=1)
                    if self.menu_state_menu==7:#M_main_menu.m_player:
                        self.set_pending_text_presses_count(2,True)
                        self.env.append_multiple_menu_contents(0,1,True,self.get_player_text(),clear_content=True)
                    elif self.menu_state_menu==8:#M_main_menu.m_save:
                        self.set_pending_text_presses_count(5,True)
                        self.env.append_multiple_menu_contents(0,1,True,["Save"],clear_content=True)
                    elif self.menu_state_menu==9:#M_main_menu.m_settings:
                        self.menu_state_cursor_y=0
                        self.menu_state_cursor_x=0
                        self.env.append_multiple_menu_contents(0,1,True,["Settings"],clear_content=True)
    def menu_tracker(self,act:int)->None:
        """Logic of tracker menu."""
        if self.menu_state_sub_menu==M_tracker.tracker:
            if act==M_actions.down:
                self.move_cursor_roll(1,self.env.game_state["tracker_flags"].shape[0],-2)
                self.env.append_multiple_menu_contents(0,1,True,self.get_tracker_text(),clear_content=True)
            elif act==M_actions.up:
                self.move_cursor_roll(-1,self.env.game_state["tracker_flags"].shape[0],-2)
                self.env.append_multiple_menu_contents(0,1,True,self.get_tracker_text(),clear_content=True)
            elif act==M_actions.left:
                self.move_cursor_roll(-self.menu_state_cursor_roll_max,self.env.game_state["tracker_flags"].shape[0],-2)
                self.env.append_multiple_menu_contents(0,1,True,self.get_tracker_text(),clear_content=True)
            if act==M_actions.right:
                self.move_cursor_roll(self.menu_state_cursor_roll_max,self.env.game_state["tracker_flags"].shape[0],-2)
                self.env.append_multiple_menu_contents(0,1,True,self.get_tracker_text(),clear_content=True)
            elif act==M_actions.b:
                self.return_to_main_menu(1)
            elif act==M_actions.a:
                if self.env.game_state["tracker_flags"][self.get_cursor_roll_sum()]>0:
                    self.set_submenu(M_tracker.side_menu)
                    self.menu_state_cursor_x=0
                    self.env.set_new_menu_layer(3,6,*self.env.centered_screen_size.tolist(),
                        y_cursor=4,x_cursor=6,vertical=True,sticky_text=True,clear_until_depth=2)
                    self.env.append_multiple_menu_contents(4,7,True,["Data","Audio","Area","Quit"])
        elif self.menu_state_sub_menu==M_tracker.side_menu:
            if act==M_actions.down:
                self.move_cursor("cursor_x",1,4,-1)
            elif act==M_actions.up:
                self.move_cursor("cursor_x",-1,4,-1)
            elif act==M_actions.b:
                self.set_submenu(M_tracker.tracker,2)
            elif act==M_actions.a:
                if self.menu_state_cursor_x==0:
                    self.set_submenu(M_tracker.data)
                    self.env.set_new_menu_layer(0,0,*self.env.centered_screen_size.tolist(),displayed_cursor=False,clear_until_depth=3)
                    idx=self.get_cursor_roll_sum()+1
                    self.env.append_multiple_menu_contents(0,1,True,["Tracker Data",f"{idx:03d} {creatures_names[idx]}"],clear_content=True)
                    with_info=self.env.game_state["tracker_flags"][idx]>1
                    if with_info:
                        self.env.append_multiple_menu_contents(3,1,True,["Was in party."])
                    self.env.set_text("Info of the\ncreature.\nMore details." if with_info else "Unknown tracker\ncreature info.")
                    self.set_pending_text_presses_count(2 if with_info>1 else 1,True)
                elif self.menu_state_cursor_x==2:
                    self.set_submenu(M_tracker.area)
                    self.env.set_new_menu_layer(0,0,*self.env.centered_screen_size.tolist(),displayed_cursor=False,clear_until_depth=3)
                    self.env.append_multiple_menu_contents(0,1,True,["Tracker Area"],clear_content=True)
                elif self.menu_state_cursor_x==3:
                    self.return_to_main_menu(1)
        elif self.menu_state_sub_menu==M_tracker.data:
            if self.text_press_routine(act):
                self.set_submenu(M_tracker.tracker,2)
        elif self.menu_state_sub_menu==M_tracker.area:
            if act in {M_actions.a,M_actions.b}:
                self.set_submenu(M_tracker.tracker,2)
        elif self.text_press_routine(act):
            self.set_submenu(M_tracker.tracker,2)
    def menu_party(self,act:int)->None:
        """Logic of party menu."""
        if self.menu_state_sub_menu==M_party.party:
            if act==M_actions.down:
                if self.move_cursor_infinite("cursor_y",1,self.env.game_state["party_size"]):
                    self.save_cursor_creature("cursor_y")
            elif act==M_actions.up:
                if self.move_cursor_infinite("cursor_y",-1,self.env.game_state["party_size"]):
                    self.save_cursor_creature("cursor_y")
            elif act==M_actions.b:
                self.return_to_main_menu(1)
            elif act==M_actions.a:
                if self.env.game_state["party_size"]==0:
                    self.return_to_main_menu(1)
                else:
                    self.set_submenu(M_party.menu_selection)
                    self.menu_state_cursor_x=0
                    creature_field_moves_str=[moves_list[k].title() for k in self.env.get_creature_field_moves(self.menu_state_cursor_y)]
                    entries=creature_field_moves_str+["Stats","Switch","Cancel"]
                    y_off=-len(entries)-1
                    self.env.set_new_menu_layer(y_off-1,3,*self.env.centered_screen_size.tolist(),
                        y_cursor=y_off,x_cursor=3,vertical=True,value=self.menu_state_cursor_x,clear_until_depth=2)
                    self.env.append_multiple_menu_contents(y_off,4,True,entries)
        elif self.menu_state_sub_menu==M_party.menu_selection:
            if act==M_actions.up:
                self.move_cursor("cursor_x",-1,6,-1)
            elif act==M_actions.b:
                self.set_submenu(M_party.party,2)
            elif act in {M_actions.down,M_actions.a}:
                creature_field_moves=self.env.get_creature_field_moves(self.menu_state_cursor_y)
                if act==M_actions.down:
                    self.move_cursor("cursor_x",1,3+len(creature_field_moves),-1)
                else:
                    if self.menu_state_cursor_x<len(creature_field_moves):
                        if self.env.can_use_field_move(creature_field_moves[self.menu_state_cursor_x]):
                            self.env.use_field_move(creature_field_moves[self.menu_state_cursor_x])
                            self.return_to_text_menu()
                            self.env.set_text("Using field\nmove!")
                        else:
                            self.set_submenu(M_party.cant_field_move,2)
                            self.env.set_text("Field move can't\nbe used.")
                    elif self.menu_state_cursor_x==len(creature_field_moves):
                        if self.env.game_state["party_size"]==0:
                            self.return_to_main_menu(1)
                        else:
                            self.set_submenu(M_party.stats)
                            self.env.set_new_menu_layer(0,0,*self.env.centered_screen_size.tolist(),displayed_cursor=False,clear_until_depth=2)
                            self.env.append_multiple_menu_contents(0,1,True,["Creatue Stats"],clear_content=True)
                            self.env.append_multiple_menu_contents(2,3,True,self.get_party_stats_text(self.menu_state_cursor_creature))
                            self.env.set_text("Stats of the\ncreature.\nMoves details.")
                            self.set_pending_text_presses_count(2,True)
                    elif self.menu_state_cursor_x==len(creature_field_moves)+1:
                        if self.env.game_state["party_size"]<2:
                            self.return_to_main_menu(1)
                        else:
                            self.set_submenu(M_party.switch_selection)
                            self.menu_state_cursor_x=self.menu_state_cursor_y
                            self.env.set_new_menu_layer(0,0,0,0,
                                y_cursor=0,x_cursor=0,vertical=True,value=self.menu_state_cursor_creature,clear_until_depth=2)
                            self.env.set_text("Which creature\nto switch?")
                    elif self.menu_state_cursor_x==len(creature_field_moves)+2:
                        self.return_to_main_menu(1)
        elif self.menu_state_sub_menu==M_party.stats:
            if self.text_press_routine(act):
                self.set_submenu(M_party.party,2)
        elif self.menu_state_sub_menu==M_party.switch_selection:
            if act==M_actions.down:
                if self.move_cursor_infinite("cursor_y",1,self.env.game_state["party_size"]):
                    self.save_cursor_creature("cursor_y")
            elif act==M_actions.up:
                if self.move_cursor_infinite("cursor_y",-1,self.env.game_state["party_size"]):
                    self.save_cursor_creature("cursor_y")
            elif act in {M_actions.a,M_actions.b}:
                switched=act==M_actions.a
                if switched:
                    self.env.switch_party_creatures(self.menu_state_cursor_y,self.menu_state_cursor_x)
                self.set_submenu(M_party.party,2)
                if switched:
                    self.env.append_multiple_menu_contents(0,1,True,self.get_party_text(),clear_content=True)
                self.save_cursor_creature("cursor_y")
                self.env.set_menu_cursor_value(self.menu_state_cursor_creature)
        elif self.menu_state_sub_menu==M_party.cant_field_move:
            if self.text_press_routine(act):
                self.set_submenu(M_party.party,2)
        elif self.text_press_routine(act):
            self.set_submenu(M_party.party,2)
    def menu_bag(self,act:int)->None:
        """Logic of bag menu."""
        if self.menu_state_sub_menu==M_bag.bag:
            if act==M_actions.down:
                if self.move_cursor_roll(1,self.env.game_state["bag_size"],0):
                    self.save_cursor_bag()
                self.env.append_multiple_menu_contents(2,3,True,self.get_bag_text(),clear_content=True)
            elif act==M_actions.up:
                if self.move_cursor_roll(-1,self.env.game_state["bag_size"],0):
                    self.save_cursor_bag()
                self.env.append_multiple_menu_contents(2,3,True,self.get_bag_text(),clear_content=True)
            elif act==M_actions.b:
                self.return_to_main_menu(1)
            elif act==M_actions.a:
                item_pos = self.get_cursor_roll_sum()
                if item_pos >= self.env.game_state["bag_size"]:
                    self.return_to_main_menu(1)
                else:
                    if self.env.is_overworld_item(item_pos):
                        self.set_submenu(M_bag.use_overworld)
                        self.env.set_text("Used item!")
                        self.set_pending_text_presses_count(1,True)
                    else:
                        self.set_submenu(M_bag.use_toss)
                        self.menu_state_cursor_x=0
                        self.env.set_new_menu_layer(5,5,*self.env.centered_screen_size.tolist(),
                            y_cursor=6,x_cursor=5,vertical=True,clear_until_depth=2)
                        self.env.append_multiple_menu_contents(6,6,True,["Use","Toss"])
        elif self.menu_state_sub_menu==M_bag.use_toss:
            if act==M_actions.down:
                self.move_cursor("cursor_x",1,2,-1)
            elif act==M_actions.up:
                self.move_cursor("cursor_x",-1,2,-1)
            elif act==M_actions.b:
                self.set_submenu(M_bag.bag,2)
            elif act==M_actions.a:
                item_pos=self.get_cursor_roll_sum()
                if self.menu_state_cursor_x==0:
                    item_id=self.env.get_item_id_by_pos(item_pos)
                    if item_id<1:
                        self.return_to_main_menu(1)
                    elif self.env.can_use_item(item_id):
                        if self.env.is_repel(item_id):
                            self.set_submenu(M_bag.use_item)
                            self.set_pending_text_presses_count(1,True)
                            self.env.set_text("Used item!")
                            self.env.use_item(item_id)
                        elif self.env.is_teachable_move(item_id):
                            self.set_submenu(M_bag.move_yesno)
                            learned_move_name=self.env.learnable_move_item_to_move_name(self.env.get_item_id_by_pos(self.get_cursor_roll_sum())).title()
                            self.env.set_text(f"This item can\nteach a move!\nThe move is\n{learned_move_name}\nTeach it to\na creature?")
                            self.set_pending_text_presses_count(2,False)
                            self.menu_state_cursor_yesno=0
                        elif self.env.is_key_item(item_id):
                            self.set_submenu(M_bag.use_overworld)
                            self.env.set_text("Used key_item!")
                            self.set_pending_text_presses_count(3,True)
                            self.env.use_item(item_id)
                        else:
                            self.set_submenu(M_bag.use_creature_selection)
                            self.menu_state_cursor_x=self.menu_state_cursor_creature
                            self.env.set_new_menu_layer(0,0,*self.env.centered_screen_size.tolist(),
                                y_cursor=0,x_cursor=0,vertical=True,value=self.menu_state_cursor_creature,clear_until_depth=3)
                            self.env.set_text("Which creature\nuse the item on?")
                            self.env.append_multiple_menu_contents(0,1,True,self.get_party_text())
                    else:
                        self.set_submenu(M_bag.cant_use)
                        self.env.set_text("Item can't be\nused!")
                        self.set_pending_text_presses_count(2,True)
                else:
                    item_pos=self.get_cursor_roll_sum()
                    if self.env.can_toss_item_by_pos(item_pos):
                        self.set_submenu(M_bag.toss_count)
                        self.menu_state_cursor_x=1
                        self.env.set_new_menu_layer(4,6,7,self.env.centered_screen_size[1],
                            y_cursor=5,x_cursor=6,vertical=True,clear_until_depth=3)
                        self.env.append_multiple_menu_contents(5,7,True,["x01"])
                    else:
                        self.set_submenu(M_bag.cant_toss)
                        self.menu_state_cursor_x=0
                        self.env.set_text("Item can't be\ntossed!")
                        self.set_pending_text_presses_count(1,True)
        elif self.menu_state_sub_menu==M_bag.cant_toss:
            if self.text_press_routine(act):
                self.set_submenu(M_bag.bag,2)
        elif self.menu_state_sub_menu==M_bag.toss_count:
            if act in {M_actions.down,M_actions.up,M_actions.a}:
                if act!=M_actions.a:
                    item_quantity=self.env.get_item_bag_quantity_by_pos(self.get_cursor_roll_sum())
                    self.move_cursor_infinite("cursor_x",1 if act==M_actions.down else -1,item_quantity,1)
                    self.env.append_multiple_menu_contents(5,7,True,[f"x{self.menu_state_cursor_x:02d}"])
                    self.env.set_menu_cursor_value(0)
                else:
                    self.set_submenu(M_bag.toss_yesno)
                    self.env.set_text("Are you sure to\ntoss this item?")
                    self.set_pending_text_presses_count(1,False)
                    self.menu_state_cursor_yesno=0
            elif act==M_actions.b:
                self.set_submenu(M_bag.bag,2)
        elif self.menu_state_sub_menu==M_bag.toss_yesno:
            if self.menu_state_pending_text_presses_count>0:
                if self.text_press_routine(act):
                    self.menu_state_cursor_yesno=0
                    self.env.set_new_menu_layer(2,5,6,self.env.centered_screen_size[1],
                        y_cursor=3,x_cursor=5,vertical=True,sticky_text=True,clear_until_depth=4)
                    self.env.append_multiple_menu_contents(3,6,True,["Yes","No"])
            else:
                if act==M_actions.down:
                    self.move_cursor("cursor_yesno",1,2,-1)
                elif act==M_actions.up:
                    self.move_cursor("cursor_yesno",-1,2,-1)
                elif act==M_actions.b:
                    self.set_submenu(M_bag.bag,2)
                elif act==M_actions.a:
                    if self.menu_state_cursor_yesno==0:
                        self.set_submenu(M_bag.tossed)
                        self.set_pending_text_presses_count(1,True)
                        self.env.drop_item_by_pos(self.get_cursor_roll_sum(),self.menu_state_cursor_x)
                        self.env.set_text("Item tossed.")
                    else:
                        self.set_submenu(M_bag.bag,2)
        elif self.menu_state_sub_menu==M_bag.tossed:
            if self.text_press_routine(act):
                self.return_to_bag_menu()
        elif self.menu_state_sub_menu==M_bag.cant_toss:
            if self.text_press_routine(act):
                self.set_submenu(M_bag.bag,2)
        elif self.menu_state_sub_menu==M_bag.use_overworld:
            if self.text_press_routine(act):
                self.return_to_overworld()
        elif self.menu_state_sub_menu==M_bag.use_item:
            if self.text_press_routine(act):
                self.set_submenu(M_bag.bag,2)
        elif self.menu_state_sub_menu==M_bag.use_creature_selection:
            if act==M_actions.down:
                if self.move_cursor_infinite("cursor_x",1,self.env.game_state["party_size"]):
                    self.save_cursor_creature("cursor_x")
            elif act==M_actions.up:
                if self.move_cursor_infinite("cursor_x",-1,self.env.game_state["party_size"]):
                    self.save_cursor_creature("cursor_x")
            elif act==M_actions.b:
                self.set_submenu(M_bag.bag,2)
            elif act==M_actions.a:
                item_pos=self.get_cursor_roll_sum()
                if self.env.can_use_item_on_creature(item_pos,self.menu_state_cursor_x):
                    if self.env.is_pp_cure(item_pos):
                        self.set_submenu(M_bag.move_selection)
                        self.menu_state_cursor_x=0
                        self.env.set_text("What move to\nselect?")
                        self.env.set_new_menu_layer(1,2,-3,self.env.centered_screen_size[1],
                            y_cursor=2,x_cursor=2,vertical=True,sticky_text=True,clear_until_depth=5)
                        self.env.append_multiple_menu_contents(2,3,True,self.get_party_moves_text(self.menu_state_cursor_creature))
                    else:
                        self.set_submenu(M_bag.used_on_creature)
                        self.env.set_text("Item used on\ncreature!")
                        self.set_pending_text_presses_count(1,True)
                        self.env.use_item_on_creature(item_pos,self.menu_state_cursor_x)
                else:
                    self.set_submenu(M_bag.cant_on_creature)
                    self.env.set_text("Can't use item\non creature!")
                    self.set_pending_text_presses_count(1,True)
        elif self.menu_state_sub_menu==M_bag.used_on_creature:
            if self.text_press_routine(act):
                self.set_submenu(M_bag.bag,2)
        elif self.menu_state_sub_menu==M_bag.cant_on_creature:
            if self.text_press_routine(act):
                self.set_submenu(M_bag.bag,2)
        elif self.menu_state_sub_menu==M_bag.lv_up_item:
            pass
        elif self.menu_state_sub_menu==M_bag.evolution_item:
            pass
        elif self.menu_state_sub_menu==M_bag.move_selection:
            if act in {M_actions.down,M_actions.up}:
                moves_count=self.env.get_creature_moves_count(self.menu_state_cursor_creature)
                self.move_cursor("cursor_x",1 if act==M_actions.down else -1,moves_count,-1)
            elif act==M_actions.b:
                self.set_submenu(M_bag.use_creature_selection,4)
                self.env.set_text("Which creature\nuse the item on?")
                self.menu_state_cursor_x=self.menu_state_cursor_creature
            elif act==M_actions.a:
                item_pos=self.get_cursor_roll_sum()
                if self.env.item_used_on_creature(item_pos,self.menu_state_cursor_creature):
                    self.set_submenu(M_bag.used_on_creature)
                    self.env.set_text("Item used on\ncreature!")
                    self.set_pending_text_presses_count(1,True)
                    self.env.use_item_on_move(item_pos,self.menu_state_cursor_creature,self.menu_state_cursor_x)
                else:
                    self.set_submenu(M_bag.cant_on_creature)
                    self.env.set_text("Can't use item\non creature!")
                    self.set_pending_text_presses_count(1,True)
        elif self.menu_state_sub_menu==M_bag.used_on_move:
            if self.text_press_routine(act):
                self.set_submenu(M_bag.bag,2)
        elif self.menu_state_sub_menu==M_bag.move_yesno:
            if self.menu_state_pending_text_presses_count>0:
                if self.text_press_routine(act):
                    self.menu_state_cursor_yesno=0
                    self.env.set_new_menu_layer(2,5,6,self.env.centered_screen_size[1],
                        y_cursor=3,x_cursor=5,vertical=True,sticky_text=True,clear_until_depth=3)
                    self.env.append_multiple_menu_contents(3,6,True,["Yes","No"])
            else:
                if act==M_actions.down:
                    self.move_cursor("cursor_yesno",1,2,-1)
                elif act==M_actions.up:
                    self.move_cursor("cursor_yesno",-1,2,-1)
                elif act==M_actions.b:
                    self.set_submenu(M_bag.bag,2)
                elif act==M_actions.a:
                    if self.menu_state_cursor_yesno==0:
                        self.set_submenu(M_bag.move_creature_selection)
                        self.menu_state_cursor_x=self.menu_state_cursor_creature
                        self.env.set_new_menu_layer(0,0,*self.env.centered_screen_size.tolist(),
                            y_cursor=0,x_cursor=0,vertical=True,value=self.menu_state_cursor_creature,clear_until_depth=4)
                        self.env.set_text("Which creature\nuse the item on?")
                        self.env.append_multiple_menu_contents(0,1,True,self.get_party_text())
                    else:
                        self.set_submenu(M_bag.bag,2)
        elif self.menu_state_sub_menu==M_bag.move_creature_selection:
            if act==M_actions.down:
                if self.move_cursor_infinite("cursor_x",1,self.env.game_state["party_size"]):
                    self.save_cursor_creature("cursor_x")
            elif act==M_actions.up:
                if self.move_cursor_infinite("cursor_x",-1,self.env.game_state["party_size"]):
                    self.save_cursor_creature("cursor_x")
            elif act==M_actions.b:
                self.set_submenu(M_bag.bag,2)
            elif act==M_actions.a:
                item_pos=self.get_cursor_roll_sum()
                if self.env.can_use_move_on_creature(item_pos,self.menu_state_cursor_x):
                    if self.env.is_duplicate_move_on_creature(item_pos,self.menu_state_cursor_x):
                        self.set_submenu(M_bag.duplicate_move)
                        self.env.set_text("Creature already\nknows the move!")
                        self.set_pending_text_presses_count(1,True)
                    elif self.env.creature_has_free_move_slot(self.menu_state_cursor_x):
                        self.set_submenu(M_bag.teached_move_new)
                        self.env.set_text("Move learned!")
                        self.set_pending_text_presses_count(1,True)
                    else:
                        self.set_submenu(M_bag.teaching_move_yesno)
                        learned_move_name=self.env.learnable_move_item_to_move_name(self.env.get_item_id_by_pos(item_pos)).title()
                        self.env.set_text(f"This creature\nwants to learn\n{learned_move_name}\nfrom the item.\nUnfortunately,\nyou must drop an\nold one to make\nroom for it.\nDo you want\nto do it?")
                        self.set_pending_text_presses_count(5,False)
                        self.menu_state_cursor_yesno=0
                else:
                    self.set_submenu(M_bag.cant_learn_move)
                    self.env.set_text("This creature\nisn't compatible\nwith the\nselected move.\nItem can't be\nused!")
                    self.set_pending_text_presses_count(3,True)
        elif self.menu_state_sub_menu==M_bag.cant_learn_move:
            if self.text_press_routine(act):
                self.set_submenu(M_bag.move_creature_selection)
                self.menu_state_cursor_x=self.menu_state_cursor_creature
                self.env.set_text("Which creature\nuse the item on?")
        elif self.menu_state_sub_menu==M_bag.duplicate_move:
            if self.text_press_routine(act):
                self.set_submenu(M_bag.move_creature_selection)
                self.menu_state_cursor_x=self.menu_state_cursor_creature
        elif self.menu_state_sub_menu==M_bag.teached_move_new:
            if self.text_press_routine(act):
                item_pos=self.get_cursor_roll_sum()
                item_id=self.env.get_item_id_by_pos(item_pos)
                if not self.env.is_key_item(item_id):
                    self.env.drop_item_by_pos(item_pos,1)
                self.return_to_bag_menu()
        elif self.menu_state_sub_menu==M_bag.teaching_move_yesno:
            if self.menu_state_pending_text_presses_count>0:
                if self.text_press_routine(act):
                    self.menu_state_cursor_yesno=0
                    self.env.set_new_menu_layer(2,5,6,self.env.centered_screen_size[1],
                        y_cursor=3,x_cursor=5,vertical=True,sticky_text=True,clear_until_depth=5)
                    self.env.append_multiple_menu_contents(3,6,True,["Yes","No"])
            else:
                if act==M_actions.down:
                    self.move_cursor("cursor_yesno",1,2,-1)
                elif act==M_actions.up:
                    self.move_cursor("cursor_yesno",-1,2,-1)
                elif act==M_actions.b:
                    self.set_submenu(M_bag.teaching_quit_yesno)
                    self.menu_state_cursor_yesno=0
                    self.env.set_text("Abort learning\nthe move?")
                    self.env.set_new_menu_layer(2,5,6,self.env.centered_screen_size[1],
                        y_cursor=3,x_cursor=5,vertical=True,sticky_text=True,clear_until_depth=5)
                    self.env.append_multiple_menu_contents(3,6,True,["Yes","No"])
                elif act==M_actions.a:
                    if self.menu_state_cursor_yesno==0:
                        self.set_submenu(M_bag.teaching_move)
                        self.menu_state_cursor_x=0
                        self.env.set_text("What move to\nforget?")
                        self.env.set_new_menu_layer(1,2,-3,self.env.centered_screen_size[1],
                            y_cursor=2,x_cursor=2,vertical=True,sticky_text=True,clear_until_depth=5)
                        self.env.append_multiple_menu_contents(2,3,True,self.get_party_moves_text(self.menu_state_cursor_creature))
                    else:
                        self.set_submenu(M_bag.teaching_quit_yesno)
                        self.menu_state_cursor_yesno=0
                        self.env.set_text("Abort learning\nthe move?")
                        self.env.set_new_menu_layer(2,5,6,self.env.centered_screen_size[1],
                            y_cursor=3,x_cursor=5,vertical=True,sticky_text=True,clear_until_depth=5)
                        self.env.append_multiple_menu_contents(3,6,True,["Yes","No"])
        elif self.menu_state_sub_menu==M_bag.teaching_quit_yesno:
            if self.menu_state_pending_text_presses_count>0:
                if self.text_press_routine(act):
                    self.menu_state_cursor_yesno=0
            else:
                if act==M_actions.down:
                    self.move_cursor("cursor_yesno",1,2,-1)
                elif act==M_actions.up:
                    self.move_cursor("cursor_yesno",-1,2,-1)
                elif act==M_actions.b:
                    self.set_submenu(M_bag.teaching_move_yesno)
                    self.env.clear_menu_content(5)
                    learned_move_name=self.env.learnable_move_item_to_move_name(self.env.get_item_id_by_pos(self.get_cursor_roll_sum())).title()
                    self.env.set_text(f"This creature\nwants to learn\n{learned_move_name}\nfrom the item.\nUnfortunately,\nyou must drop an\nold one to make\nroom for it.\nDo you want\nto do it?")
                    self.set_pending_text_presses_count(5,False)
                    self.menu_state_cursor_yesno=0
                elif act==M_actions.a:
                    if self.menu_state_cursor_yesno==0:
                        self.set_submenu(M_bag.teaching_move_quit)
                        self.env.set_text("No moves was\nlearned!\nNo item was\nused!")
                        self.set_pending_text_presses_count(2,True)
                    else:
                        self.set_submenu(M_bag.teaching_move_yesno)
                        learned_move_name=self.env.learnable_move_item_to_move_name(self.env.get_item_id_by_pos(self.get_cursor_roll_sum())).title()
                        self.env.set_text(f"This creature\nwants to learn\n{learned_move_name}\nfrom the item.\nUnfortunately,\nyou must drop an\nold one to make\nroom for it.\nDo you want\nto do it?")
                        self.set_pending_text_presses_count(5,False)
                        self.menu_state_cursor_yesno=0
        elif self.menu_state_sub_menu==M_bag.teaching_move_quit:
            if self.text_press_routine(act):
                self.set_submenu(M_bag.bag,2)
        elif self.menu_state_sub_menu==M_bag.teaching_move:
            if act in {M_actions.down,M_actions.up}:
                moves_count=self.env.get_creature_moves_count(self.menu_state_cursor_creature)
                self.move_cursor("cursor_x",1 if act==M_actions.down else -1,moves_count,-1)
            elif act==M_actions.b:
                self.set_submenu(M_bag.teaching_quit_yesno)
                self.menu_state_cursor_yesno=0
                self.env.set_text("Abort learning\nthe move?")
                self.env.set_new_menu_layer(2,5,6,self.env.centered_screen_size[1],
                    y_cursor=3,x_cursor=5,vertical=True,sticky_text=True,clear_until_depth=5)
                self.env.append_multiple_menu_contents(3,6,True,["Yes","No"])
            elif act==M_actions.a:
                item_pos=self.get_cursor_roll_sum()
                item_id=self.env.get_item_id_by_pos(item_pos)
                if self.env.move_forgettable_by_creature(self.menu_state_cursor_creature,self.get_cursor_roll_sum(),self.menu_state_cursor_x):
                    self.set_submenu(M_bag.teached_move_replaced)
                    self.env.clear_menu_content(5)
                    learned_move_id=self.env.learnable_move_item_to_move_id(item_id)
                    learned_move_name=self.env.learnable_move_item_to_move_name(item_id).title()
                    self.env.set_text(f"The creature\nis forgetting\nthe old move.\nIt concentrates\non the used\nitem and...\nIt learns\n{learned_move_name}!")
                    self.set_pending_text_presses_count(4,True)
                    self.env.learn_move_to_creature(self.menu_state_cursor_creature,learned_move_id,self.menu_state_cursor_x)
                else:
                    self.set_submenu(M_bag.cant_delete_move)
                    self.env.clear_menu_content(5)
                    self.env.set_text("This move can't\nbe removed.")
                    self.set_pending_text_presses_count(1,True)
        elif self.menu_state_sub_menu==M_bag.cant_delete_move:
            if self.text_press_routine(act):
                self.set_submenu(M_bag.teaching_move)
                self.menu_state_cursor_x=0
                self.env.set_text("What move to\nforget?")
                self.env.set_new_menu_layer(1,2,-3,self.env.centered_screen_size[1],
                    y_cursor=2,x_cursor=2,vertical=True,sticky_text=True,clear_until_depth=5)
                self.env.append_multiple_menu_contents(2,3,True,self.get_party_moves_text(self.menu_state_cursor_creature))
        elif self.menu_state_sub_menu==M_bag.teached_move_replaced:
            if self.text_press_routine(act):
                item_pos=self.get_cursor_roll_sum()
                item_id=self.env.get_item_id_by_pos(item_pos)
                if not self.env.is_key_item(item_id):
                    self.env.drop_item_by_pos(item_pos,1)
                self.return_to_bag_menu()
        elif self.text_press_routine(act):
            self.set_submenu(M_bag.bag,2)
    def menu_player(self,act:int)->None:
        """Logic of player menu."""
        if act in {M_actions.a,M_actions.b}:
            self.return_to_main_menu(1)
    def menu_save(self,act:int)->None:
        """Logic of save menu."""
        if self.text_press_routine(act):
            self.return_to_main_menu(1)
    def menu_settings(self,act:int)->None:
        """Logic of settings menu."""
        if act in {M_actions.b,M_actions.start}:
            self.return_to_main_menu(1)
    def menu_pc(self,act:int)->None:
        """Logic of pc menu."""
        return
    def menu_battle(self,act:int)->None:
        """Logic of battle menu."""
        return
    def repr(self,action:int=0xFF)->None:
        """Prints actions for debug purposes."""
        if not self.header_printed:
            print("Act\tMenu\t\tSubmenu\t\tMain\tY\tY-roll\tX\tPress")
            self.header_printed=True
        menu_name=self.repr_menu_enum.get(self.menu_state_menu,M_main_menu.m_unknown.name)
        submenu_name=self.repr_submenu_enum.get(self.menu_state_menu,{}).get(self.menu_state_sub_menu,f"Sub({self.menu_state_sub_menu:d})")
        act=self.str_to_action(action)
        print("{}\t{:11}\t{:15}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}".format("*" if act<0 else act,menu_name,submenu_name,*[getattr(self,f"menu_state_{k}") for k in ["main_menu_cursor","cursor_y","cursor_roll_y","cursor_x","pending_text_presses_count"]]))
    def step_menu(self,action:int=0,debug:bool=False)->None:
        """Main logic for menu interactions."""
        if self.menu_state_menu<2 or self.menu_state_menu>9:#M_main_menu.m_settings:
            self.no_menu_overworld(action)
        elif self.menu_state_menu==2:#M_main_menu.m_text:
            self.menu_text(action)
        elif self.menu_state_menu==3:#M_main_menu.m_menu:
            self.menu_start(action)
        elif self.menu_state_menu==4:#M_main_menu.m_tracker:
            self.menu_tracker(action)
        elif self.menu_state_menu==5:#M_main_menu.m_party:
            self.menu_party(action)
        elif self.menu_state_menu==6:#M_main_menu.m_bag:
            self.menu_bag(action)
        elif self.menu_state_menu==7:#M_main_menu.m_player:
            self.menu_player(action)
        elif self.menu_state_menu==8:#M_main_menu.m_save:
            self.menu_save(action)
        elif self.menu_state_menu==9:#M_main_menu.m_settings:
            self.menu_settings(action)
        if debug:
            self.repr(action)
    def run_cli_test(self)->None:
        """Manual test routine."""
        buttons=["s","d","d","a"]+["a"]
        for b in buttons:
            self.step_menu(b)
        self.repr(0)
        while True:
            act_str=input("Input >> ")
            if len(act_str)<1:
                break
            act=self.str_to_action(act_str)
            self.step_menu(act,True)
