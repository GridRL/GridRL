#!/usr/bin/env python3

"""Basic menu implementations."""

from typing import Union
from enum import IntEnum
from copy import deepcopy
import sys
#import cython
import numpy as np

sys.dont_write_bytecode=True

if __package__ is None or len(__package__)==0:
    from core_environment import GridRLAbstractEnv
else:
    from gridrl.core_environment import GridRLAbstractEnv

__all__=["MenuBase"]

if True:#not cython.compiled:
    M_actions=IntEnum("M_actions",["down","left","right","up","a","b","start","select"],start=0)
    M_overworld=IntEnum("M_overworld",["overworld"],start=0)
    M_unknown=IntEnum("M_unknown",["unknown"],start=0)
    M_text=IntEnum("M_text",["text"],start=0)
    M_menu=IntEnum("M_menu",["menu"],start=0)

class MenuBase:
    """Base Menu class to for a game."""
    def __init__(self,env:GridRLAbstractEnv,*args,**kwargs)->None:
        """Constructor."""
        self.env=None
        self.set_env(env)
        self.force_menu=False
        self.log_screen=False
        self.menu_state_menu=np.uint8(0)
        self.menu_state_sub_menu=np.uint8(0)
        self.menu_state_main_menu_cursor=np.uint8(0)
        self.menu_state_cursor_x=np.uint8(0)
        self.menu_state_cursor_y=np.uint8(0)
        self.menu_state_cursor_roll_y=np.uint8(0)
        self.menu_state_cursor_roll_max=np.uint8(0)
        self.menu_state_cursor_yesno=np.uint8(0)
        self.menu_state_pending_text_presses_count=np.uint8(0)
    def has_env(self)->bool:
        """Check if the environment is set."""
        return self.env is not None
    def set_env(self,env:GridRLAbstractEnv,recursion_depth:int=0)->None:
        """Set the environment."""
        self.env=env
        if recursion_depth<2 and hasattr(self.env,"set_menu"):
            self.env.set_menu(self,recursion_depth+1)
    def set_force_menu(self,force_menu:bool=True)->None:
        """Set the flag to enforce menu being active, used for benchmarking."""
        self.force_menu=bool(force_menu)
    def get_extra_attribute_state_names(self)->list[str]:
        """List of extra attribute names preserved in a save state."""
        return []
    def get_attribute_state_names(self)->set:
        """List of all attribute names preserved in a save state."""
        return set(["menu_state_menu","menu_state_sub_menu","menu_state_main_menu_cursor",
            "menu_state_cursor_x","menu_state_cursor_y","menu_state_cursor_roll_y",
            "menu_state_cursor_roll_max","menu_state_cursor_yesno",
            "menu_state_pending_text_presses_count",
        ]+list(self.get_extra_attribute_state_names()))
    def load_state(self,state:dict)->bool:
        """Load from a saved state dictionary."""
        if not isinstance(state,dict):
            return self.load_state(state.load_state()) if isinstance(state,MenuBase) else False
        for k in self.get_attribute_state_names():
            if k in state and len(k)>0 and hasattr(self,k):
                setattr(self,k,deepcopy(state[k]))
        return True
    def save_state(self)->dict:
        """Save the current state to a dictionary."""
        state={k:deepcopy(getattr(self,k)) for k in self.get_attribute_state_names() if len(k)>0 and hasattr(self,k)}
        return state
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
        ],dtype=np.uint8)
    def reset(self,seed:int=None)->None:
        """Reset the menu."""
        if self.has_env() and hasattr(self.env,"game_state"):
            self.env.game_state["menu_type"]=0
            self.env.clear_menu_content(0)
            self.env.set_menu_cursor_origin(0,0,vertical=True,displayed=False)
            self.env.text_changed_during_step=True
            self.env.menu_changed_during_step=True
    def step_menu(self,action:int):
        """Main logic for menu interactions."""
        self.env.menu_changed_during_step=False
        if self.env.game_state["menu_type"]==0:
            if action==self.env.action_menu_id:
                self.env.game_state["menu_type"]=2
                self.env.clear_menu_content()
                self.env.set_menu_cursor_origin(1,-5,vertical=True)
                self.env.append_menu_content(1,-4,"Menu")
                self.env.menu_changed_during_step=True
        elif self.env.game_state["menu_type"]==2:
            if action in {self.env.action_back_id,self.env.action_menu_id}:
                self.env.game_state["menu_type"]=0
                self.env.clear_menu_content()
                self.env.set_menu_cursor_origin(0,0,vertical=True,displayed=False)
                self.env.handle_menu_actions(self.env.action_back_id)
                self.env.menu_changed_during_step=True
    def step(self,action:int)->None:
        """Wrapper for benchmark, steps the menu."""
        self.step_menu(action)
        self.env.step_count+=1
    def repr(self,action)->None:
        """Prints actions for debug purposes."""
        return
    def get_cursor_roll_sum(self)->int:
        """Return the total index targeted by rolling cursor_y."""
        return self.menu_state_cursor_y+self.menu_state_cursor_roll_y
    def move_cursor(self,cursor_name:str,offset,limit_value,limit_off)->bool:
#    def move_cursor(self,cursor_name:str,offset:int,limit_value:int=0xFF,limit_off:int=-1)->bool:
        """Update cursor position."""
        if cursor_name=="cursor_x":
            val=self.menu_state_cursor_x
        elif cursor_name=="cursor_yesno":
            val=self.menu_state_cursor_yesno
        elif cursor_name=="cursor_u":
            val=self.menu_state_cursor_y
        else:
            gs_name=f"menu_state_{cursor_name}"
            val=getattr(self,gs_name)
        if offset>=0:
            if val<limit_off+limit_value:#self.get_limit_value(limit_value):
                #self.menu_state[cursor_name]+=1
                if cursor_name=="cursor_x":
                    self.menu_state_cursor_x+=1
                    self.env.set_menu_cursor_value(self.menu_state_cursor_x)
                elif cursor_name=="cursor_yesno":
                    self.menu_state_cursor_yesno+=1
                    self.env.set_menu_cursor_value(self.menu_state_cursor_yesno)
                elif cursor_name=="cursor_u":
                    self.menu_state_cursor_y+=1
                    self.env.set_menu_cursor_value(self.menu_state_cursor_y)
                else:
                    setattr(self,gs_name,val+1)
                    self.env.set_menu_cursor_value(getattr(self,gs_name))
        else:
            if val>0:
                #self.menu_state[cursor_name]-=1
                if cursor_name=="cursor_x":
                    self.menu_state_cursor_x-=1
                    self.env.set_menu_cursor_value(self.menu_state_cursor_x)
                elif cursor_name=="cursor_yesno":
                    self.menu_state_cursor_yesno-=1
                    self.env.set_menu_cursor_value(self.menu_state_cursor_yesno)
                elif cursor_name=="cursor_u":
                    self.menu_state_cursor_y-=1
                    self.env.set_menu_cursor_value(self.menu_state_cursor_y)
                else:
                    setattr(self,gs_name,val-1)
                    self.env.set_menu_cursor_value(getattr(self,gs_name))
    def move_cursor_roll(self,offset,limit_value,limit_off)->bool:
#    def move_cursor_roll(self,offset:int,limit_value:int,limit_off:int=-1)->bool:
        """Update cursor position accounting for a rolling interval."""
        ret=False
        if offset >= 0:
            for _ in range(offset):
                if self.get_cursor_roll_sum()<limit_off+limit_value:#+self.get_limit_value(limit_value):
                    if self.menu_state_cursor_y+1>=self.menu_state_cursor_roll_max:
                        self.menu_state_cursor_roll_y+=1
                    else:
                        self.menu_state_cursor_y+=1
                        self.env.set_menu_cursor_value(self.menu_state_cursor_y)
                    ret=True
            return ret
        ret=False
        for _ in range(-offset):
            if self.get_cursor_roll_sum()>0:
                if self.menu_state_cursor_y==0:
                    self.menu_state_cursor_roll_y-=1
                else:
                    self.menu_state_cursor_y-=1
                    self.env.set_menu_cursor_value(self.menu_state_cursor_y)
                ret=True
        return ret
    def move_cursor_infinite(self,cursor_name:str,offset:int,limit_value:int,start:int=0)->bool:
        """Update cursor position returning back at the boundaries."""
        #limit_value=self.get_limit_value(limit_value)
        #self.menu_state[cursor_name]=start+((self.menu_state[cursor_name]-start+limit_value+offset)%limit_value)
        if limit_value==0:
            return False
        if cursor_name=="main_menu_cursor":
            self.menu_state_main_menu_cursor=start+((self.menu_state_main_menu_cursor-start+limit_value+offset)%limit_value)
            self.env.set_menu_cursor_value(self.menu_state_main_menu_cursor)
        elif cursor_name=="cursor_x":
            self.menu_state_cursor_x=start+((self.menu_state_cursor_x-start+limit_value+offset)%limit_value)
            self.env.set_menu_cursor_value(self.menu_state_cursor_x)
        else:
            self.menu_state_cursor_y=start+((self.menu_state_cursor_y-start+limit_value+offset)%limit_value)
            self.env.set_menu_cursor_value(self.menu_state_cursor_y)
        return True
    def set_menu(self,menu:Union[IntEnum,int],sub_menu:Union[IntEnum,int,None]=None,clear_depth:int=-1)->None:
        """Set the menu and submenu indexes."""
        self.menu_state_menu=menu.value if isinstance(menu,IntEnum) else menu
        if sub_menu is not None:
            self.set_submenu(sub_menu,-1)
        if hasattr(self.env,"game_state"):
            self.env.game_state["menu_type"]=self.menu_state_menu
        if clear_depth>-1:
            self.env.clear_menu_content(clear_depth)
    def set_submenu(self,sub_menu:Union[IntEnum,int],clear_depth:int=-1)->None:
        """Set the submenu index."""
        self.menu_state_sub_menu=sub_menu.value if isinstance(sub_menu,IntEnum) else sub_menu
        if hasattr(self.env,"game_state"):
            self.env.game_state["sub_menu_type"]=self.menu_state_sub_menu
        if clear_depth>-1:
            self.env.clear_menu_content(clear_depth)
    def set_pending_text_presses_count(self,presses,remove_one,single_line:bool=False)->None:
        """Set the amount of times the buttons must be pressed to display all text queue."""
        off=1 if remove_one else presses
        self.menu_state_pending_text_presses_count=max(0,presses-off)
        if self.has_env():
            self.menu_state_pending_text_presses_count=max(self.menu_state_pending_text_presses_count,len(self.env.text_queue)//(1 if single_line else 2))
    def set_npc_text_with_presses_count(self,text:str,presses:int=0):
        """Append text and updates the expected button presses count."""
        self.env.set_text(text)
        self.set_pending_text_presses_count(presses,False,False)
    def text_press_routine(self,act,single_line:bool=False):
        """Process the button check and scrolls the text queue."""
        if act in {self.env.action_interact_id,self.env.action_back_id}:
#        if act==M_actions.a or act==M_actions.b:
            ret=self.env.step_text(single_line=single_line,keep_end=True)
            if self.menu_state_pending_text_presses_count>0:
                self.menu_state_pending_text_presses_count-=1
            if self.menu_state_pending_text_presses_count==0:
                return ret
        return False
    def return_to_overworld(self)->None:
        """Close any menu."""
        self.set_menu(0,0,0)
        self.env.game_state["text_type"]=0
        self.env.clear_text()
    def return_to_unknown_menu(self,clear_until_depth:int=1)->None:
        """Set the menu to an unknown value. Placeholder for the [1] value, don't use it."""
        self.set_menu(1,0,clear_until_depth)
    def return_to_text_menu(self)->None:
        """Open the text prompt."""
        self.set_menu(2,0,0)
        self.env.game_state["text_type"]=1
    def return_to_main_menu(self,clear_until_depth:int=1)->None:
        """Open the main menu."""
        self.set_menu(3,0,clear_until_depth)
    def menu_text(self,act:int)->None:
        """Logic of text dialogs."""
        if self.text_press_routine(act):
            self.return_to_overworld()
