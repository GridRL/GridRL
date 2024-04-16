#!/usr/bin/env python3

"""Menu implementation of the game [exploration_world1]."""

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
    from gridrl.menu_base import MenuBase,M_actions,M_overworld,M_unknown,M_text,M_menu
    from gridrl.games.exploration_world1.constants import powerups_list,items_list
except ModuleNotFoundError:
    from core_environment import GridRLAbstractEnv
    from menu_base import MenuBase,M_actions,M_overworld,M_unknown,M_text,M_menu
    from games.exploration_world1.constants import powerups_list,items_list

sys.dont_write_bytecode=True

if True:#not cython.compiled:
    M_main_menu=IntEnum("M_main_menu",["m_overworld","m_unknown","m_text","m_menu","m_powerup","m_bag","m_player"],start=0)
    M_powerup=IntEnum("M_powerup",["powerup","menu_selection","cant_powerup"],start=0)
    M_bag=IntEnum("M_bag",["bag","use_item"],start=0)
    M_player=IntEnum("M_player",["player"],start=0)

class ExplorationWorld1Menu(MenuBase):
    """Menu class of the game exploration_world1."""
    def __init__(self,env:GridRLAbstractEnv=None,*args,**kwargs)->None:
        """Constructor."""
        super().__init__(env,*args,**kwargs)
        self.repr_menu_enum={(k.value+256)%256:k.name for k in M_main_menu}
        self.repr_submenu_enum={(em.value+256)%256:{(k.value+256)%256:k.name for k in es} for em,es in zip(M_main_menu,[M_overworld,M_unknown,M_text,M_menu,M_powerup,M_bag,M_player])}
        self.actions={"down":0,"left":1,"right":2,"up":3,"a":4,"b":5,"start":6,"select":7,
            "d":0,"l":1,"r":2,"u":3,"s":6,"z":7,
            "0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,
        }
        self.header_printed=False
        self.menu_state_cursor_powerup=np.uint8(0)
        self.menu_state_cursor_bag=np.uint8(0)
        self.menu_state_cursor_roll_bag=np.uint8(0)
        self.reset()
    def get_extra_attribute_state_names(self)->list[str]:
        """List of extra attribute names preserved in a save state."""
        return ["menu_state_cursor_powerup","menu_state_cursor_bag","menu_state_cursor_roll_bag"]
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
            self.menu_state_cursor_powerup,
            self.menu_state_cursor_bag,
            self.menu_state_cursor_roll_bag,
        ],dtype=np.uint8)
    def save_cursor_powerup(self,cursor_name:str)->None:
        """Update the powerup cursor value."""
        if cursor_name=="cursor_x":
            self.menu_state_cursor_powerup=self.menu_state_cursor_x
        else:
            self.menu_state_cursor_powerup=self.menu_state_cursor_y
    def save_cursor_bag(self)->None:
        """Update the bag cursor values."""
        self.menu_state_cursor_roll_bag=self.menu_state_cursor_roll_y
        self.menu_state_cursor_bag=self.menu_state_cursor_y
    def get_powerup_text(self):
        """Text for party menu names list."""
        powerups_flags=[self.env.get_event_flag(k) for k in powerups_list]
        if np.sum(powerups_flags)==0:
            text_list=["No powerups."]
        else:
            text_list=["----" if self.env.get_event_flag(k)<1 else k[8:] for k in powerups_list]
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
        return ["Player"]#,f"Steps: {self.env.step_count:d}"]
    def no_menu_overworld(self,act:int)->None:
        """Logic without menu active."""
        if act==M_actions.start or self.force_menu:
            self.return_to_main_menu(0)
            self.env.set_new_menu_layer(0,-5,*self.env.centered_screen_size.tolist(),
                y_cursor=1,x_cursor=-5,vertical=True,value=self.menu_state_main_menu_cursor,clear_until_depth=0)
            self.env.append_multiple_menu_contents(1,-4,True,["Powerup","Bag","Player","Exit"])
    def menu_start(self,act:int)->None:
        """Logic of main menu."""
        if act in {M_actions.b,M_actions.start}:
            self.return_to_overworld()
        elif act==M_actions.down:
            self.move_cursor_infinite("main_menu_cursor",1,4)
        elif act==M_actions.up:
            self.move_cursor_infinite("main_menu_cursor",-1,4)
        elif act==M_actions.a:
            if self.menu_state_main_menu_cursor>=3:
                self.return_to_overworld()
            else:
                self.set_menu(self.menu_state_main_menu_cursor+4,0)
                self.menu_state_pending_text_presses_count=0
                if self.menu_state_menu==4:#M_main_menu.m_powerup:
                    if self.env.get_event_flag("powerup_debush")>0:
                        self.menu_state_cursor_y=self.menu_state_cursor_powerup
                        self.env.set_new_menu_layer(0,0,*self.env.centered_screen_size.tolist(),
                            y_cursor=0,x_cursor=0,vertical=True,value=self.menu_state_cursor_powerup,clear_until_depth=1)
                        self.env.append_multiple_menu_contents(0,1,True,self.get_powerup_text())
                    else:
                        self.return_to_overworld()
                elif self.menu_state_menu==5:#M_main_menu.m_bag:
                    self.menu_state_cursor_roll_max=3
                    self.menu_state_cursor_roll_y=self.menu_state_cursor_roll_bag
                    self.menu_state_cursor_y=self.menu_state_cursor_bag
                    self.env.set_new_menu_layer(1,2,-2,self.env.centered_screen_size[1],
                        y_cursor=2,x_cursor=2,vertical=True,value=self.menu_state_cursor_bag,clear_until_depth=1)
                    self.env.append_multiple_menu_contents(2,3,True,self.get_bag_text())
                else:
                    self.env.set_new_menu_layer(0,0,*self.env.centered_screen_size.tolist(),displayed_cursor=False,clear_until_depth=1)
                    if self.menu_state_menu==6:#M_main_menu.m_player:
                        self.set_pending_text_presses_count(2,True)
                        self.env.append_multiple_menu_contents(0,1,True,self.get_player_text(),clear_content=True)
    def menu_powerup(self,act:int)->None:
        """Logic of powerup menu."""
        if self.menu_state_sub_menu==M_powerup.powerup:
            if act==M_actions.down:
                if self.move_cursor_infinite("cursor_y",1,8):
                    self.save_cursor_powerup("cursor_y")
            elif act==M_actions.up:
                if self.move_cursor_infinite("cursor_y",-1,8):
                    self.save_cursor_powerup("cursor_y")
            elif act==M_actions.b:
                self.return_to_main_menu(1)
            elif act==M_actions.a:
                powerup_id=self.env.get_powerup_id(self.menu_state_cursor_y)
                if self.env.get_event_flag(powerups_list[powerup_id])<1:
                    self.return_to_main_menu(1)
                else:
                    self.set_submenu(M_powerup.menu_selection)
                    self.menu_state_cursor_x=0
                    self.env.set_new_menu_layer(-5,3,*self.env.centered_screen_size.tolist(),
                        y_cursor=-4,x_cursor=3,vertical=True,value=self.menu_state_cursor_x,clear_until_depth=2)
                    self.env.append_multiple_menu_contents(-4,4,True,["Use","Cancel"])
        elif self.menu_state_sub_menu==M_powerup.menu_selection:
            if act==M_actions.down:
                self.move_cursor("cursor_x",1,2,-1)
            elif act==M_actions.up:
                self.move_cursor("cursor_x",-1,2,-1)
            elif act==M_actions.b:
                self.set_submenu(M_powerup.powerup,2)
            elif act==M_actions.a:
                if self.menu_state_cursor_x==0:
                    powerup_id=self.env.get_powerup_id(self.menu_state_cursor_y)
                    if self.env.can_use_powerup(powerup_id):
                        self.env.use_powerup(powerup_id)
                        self.return_to_text_menu()
                        self.env.set_text("Using\npowerup!")
                    else:
                        self.set_submenu(M_powerup.cant_powerup,2)
                        self.env.set_text("Powerup can't\nbe used.")
                else:
                    self.return_to_main_menu(1)
        elif self.menu_state_sub_menu==M_powerup.cant_powerup:
            if self.text_press_routine(act):
                self.set_submenu(M_powerup.powerup,2)
        elif self.text_press_routine(act):
            self.set_submenu(M_powerup.powerup,2)
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
#                    self.env.use_item(item_id)
                    self.return_to_text_menu()
                    self.env.set_text("Used item!")
        elif self.menu_state_sub_menu==M_bag.use_item:
            if self.text_press_routine(act):
                self.set_submenu(M_bag.bag,2)
        elif self.text_press_routine(act):
            self.set_submenu(M_bag.bag,2)
    def menu_player(self,act:int)->None:
        """Logic of player menu."""
        if act in {M_actions.a,M_actions.b}:
            self.return_to_main_menu(1)
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
        elif self.menu_state_menu==4:#M_main_menu.m_powerup:
            self.menu_powerup(action)
        elif self.menu_state_menu==5:#M_main_menu.m_bag:
            self.menu_bag(action)
        elif self.menu_state_menu==6:#M_main_menu.m_player:
            self.menu_player(action)
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
