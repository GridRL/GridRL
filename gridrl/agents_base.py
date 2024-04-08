#!/usr/bin/env python3

"""Basic agents implementations."""

import sys
import numpy as np
sys.dont_write_bytecode=True

if __package__ is None or len(__package__)==0:
    from core_environment import GridRLAbstractEnv
else:
    from gridrl.core_environment import GridRLAbstractEnv

class AgentBase:
    """Base Agent class to sent actions to the environment."""
    def __init__(self,env:GridRLAbstractEnv,*args,**kwargs)->None:
        """Constructor."""
        self.env=None
        self.scheduled_actions=[]
        self.set_env(env)
        self.flags_count=np.sum(self.env.game_state["event_flags"]) if self.has_env() else 0
    def has_env(self)->bool:
        """Check if the environment is set."""
        return self.env is not None
    def set_env(self,env:GridRLAbstractEnv,recursion_depth:int=0)->None:
        """Set the environment."""
        self.env=env
        if recursion_depth<2 and hasattr(self.env,"set_agent"):
            self.env.set_agent(self,recursion_depth+1)
    def reset(self)->None:
        """Reset the agent."""
        return
    def pre_roll_initialize(self)->None:
        """Run before starting the roll."""
        self.reset_scheduled_actions()
    def lookup_str_to_env_button(self,button:str)->int:
        """Converts string buttons to integer actions."""
        return button if isinstance(button,int) else self.env.input_env_map[button]
    def reset_scheduled_actions(self)->None:
        """Clear the scheduled predicted actions."""
        self.scheduled_actions.clear()
    def set_scheduled_actions(self,actions:list,skip_invert:bool=False)->None:
        """Set the scheduled predicted actions."""
        self.scheduled_actions=list(actions[::None if skip_invert else -1])
    def set_scheduled_action_teleport(self,map_id:int)->None:
        """Experimental - set action values for the legit teleport."""
        if not hasattr(self.env,"action_teleport_id"):
            return
        self.env.set_secondary_action_value(map_id)
        self.set_scheduled_actions([self.env.action_teleport_id])
    def pop_last_scheduled_action(self)->int:
        """Pop from the scheduled action predictions."""
        return 0 if len(self.scheduled_actions)==0 else self.scheduled_actions.pop()
    def predict_action(self)->int:
        """Get the next predicted action."""
        return self.env.roll_random_actions_without_nop(1)[0]
    def step_game(self,action:int)->int:
        """Step the game without processing observations."""
        self.env.step_game(action)
        self.env.update_reward(action)
        return 0
    def step(self,action:int)->tuple:
        """Step the environment."""
        return self.env.step(action)
    def check_early_stopping(self)->bool:
        """Early stopping inside roll for a new event flag reached."""
        fc=self.env.get_events_flag_sum()
        ret=fc>self.flags_count
        self.flags_count=fc
        return ret
    def run_roll_game(self,roll_steps:int,early_stopping:bool=False)->np.ndarray:
        """Run the step roll without processing observations."""
        self.pre_roll_initialize()
        actions=np.full((roll_steps,),self.env.action_nop_id,dtype=np.int16,order="C")
        for i in range(roll_steps-1):
            act=self.predict_action()
            self.step_game(act)
            actions[i]=act
            if early_stopping and self.check_early_stopping():
                break
        act=self.predict_action()
        self.step(act)
        actions[i+1]=act
        return actions[:i+2]
    def run_roll(self,roll_steps:int,early_stopping:bool=False)->np.ndarray:
        """Run the step roll."""
        self.pre_roll_initialize()
        actions=np.full((roll_steps,),self.env.action_nop_id,dtype=np.int16,order="C")
        for i in range(roll_steps):
            act=self.predict_action()
            data=self.step(act)
            actions[i]=act
            if data[2] or (early_stopping and self.check_early_stopping()):
                break
        return actions[:i+1]
    def hook_before_warp(self,global_warped:bool,movements:list)->None:
        """Game Hook: executed before entering a warp."""
        return
    def hook_after_warp(self)->None:
        """Game Hook: executed after exiting a warp."""
        return
