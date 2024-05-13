#!/usr/bin/env python3

"""Basic agents implementations."""

import sys
import numpy as np
sys.dont_write_bytecode=True

if __package__ is None or len(__package__)==0:
    from core_environment import GridRLAbstractEnv
else:
    from gridrl.core_environment import GridRLAbstractEnv

__all__=["AgentBase"]

class AgentBase:
    """Base Agent class to sent actions to the environment."""
    def __init__(self,env:GridRLAbstractEnv,*args,**kwargs)->None:
        """Constructor."""
        self.env=None
        self.game_env=None
        self.scheduled_actions=[]
        self.set_env(env)
        self.flags_count=np.sum(self.game_env.game_state["event_flags"]) if self.has_env() else 0
    def has_env(self)->bool:
        """Check if the environment is set."""
        return self.game_env is not None
    def set_env(self,env:GridRLAbstractEnv,recursion_depth:int=0)->None:
        """Set the environment."""
        self.game_env=env
        if recursion_depth<2 and hasattr(self.game_env,"set_agent"):
            self.env=env
            self.game_env.set_agent(self,recursion_depth+1)
    def get_extra_attribute_state_names(self)->list[str]:
        """List of extra attribute names preserved in a save state."""
        return []
    def get_attribute_state_names(self)->set:
        """List of all attribute names preserved in a save state."""
        return set(["flags_count","scheduled_actions",
            ]+list(self.get_extra_attribute_state_names()))
    def load_state(self,state:dict)->bool:
        """Load from a saved state dictionary."""
        if not isinstance(state,dict):
            return self.load_state(state.load_state()) if isinstance(state,AgentBase) else False
        for k in self.get_attribute_state_names():
            if k in state and len(k)>0 and hasattr(self,k):
                setattr(self,k,deepcopy(state[k]))
        return True
    def save_state(self)->dict:
        """Save the current state to a dictionary."""
        state={k:deepcopy(getattr(self,k)) for k in self.get_attribute_state_names() if len(k)>0 and hasattr(self,k) and not k in ["env","game_env"]}
        return state
    def reset(self)->None:
        """Reset the agent."""
        return
    def pre_roll_initialize(self)->None:
        """Run before starting the roll."""
        self.reset_scheduled_actions()
    def lookup_str_to_env_button(self,button:str)->int:
        """Converts string buttons to integer actions."""
        return button if isinstance(button,int) else self.game_env.input_env_map[button]
    def reset_scheduled_actions(self)->None:
        """Clear the scheduled predicted actions."""
        self.scheduled_actions.clear()
    def extend_scheduled_actions(self,actions:list,skip_invert:bool=False)->None:
        """Extend the scheduled predicted actions."""
        self.scheduled_actions.extend(list(actions[::None if skip_invert else -1]))
    def set_scheduled_actions(self,actions:list,skip_invert:bool=False)->None:
        """Set the scheduled predicted actions."""
        self.scheduled_actions=list(actions[::None if skip_invert else -1])
    def set_scheduled_action_teleport(self,map_id:int)->bool:
        """Experimental - set action values for the legit teleport."""
        if not hasattr(self.game_env,"action_teleport_id"):
            return False
        ret=self.game_env.set_secondary_action_value(map_id)
        if ret:
            self.set_scheduled_actions([self.game_env.action_teleport_id])
        return ret
    def pop_last_scheduled_action(self)->int:
        """Pop from the scheduled action predictions."""
        return 0 if len(self.scheduled_actions)==0 else self.scheduled_actions.pop()
    def predict_action(self)->int:
        """Get the next predicted action."""
        return self.game_env.roll_random_actions_without_nop(1)[0]
    def step_game(self,action:int)->int:
        """Step the game without processing observations."""
        self.game_env.step_game(action)
        self.game_env.update_reward(action)
        return 0
    def step(self,action:int)->tuple:
        """Step the environment."""
        return self.env.step(action)
    def check_early_stopping(self)->bool:
        """Early stopping inside roll for a new event flag reached."""
        fc=self.game_env.get_event_flags_sum()
        ret=fc>self.flags_count
        self.flags_count=fc
        return ret
    def run_roll_game_from_actions(self,actions:np.ndarray,early_stopping:bool=False)->np.ndarray:
        """Run the step roll from actions without processing observations."""
        for i in range(len(actions)-1):
            self.step_game(actions[i])
            if self.game_env.steps_or_game_completed() or (early_stopping and self.check_early_stopping()):
                break
        self.step(actions[i+1])
        return i+2
    def run_roll_game(self,roll_steps:int,early_stopping:bool=False)->np.ndarray:
        """Run the step roll without processing observations."""
        self.pre_roll_initialize()
        actions=np.full((roll_steps,),self.game_env.action_nop_id,dtype=np.int16,order="C")
        for i in range(roll_steps-1):
            act=self.predict_action()
            self.step_game(act)
            actions[i]=act
            if self.game_env.steps_or_game_completed() or (early_stopping and self.check_early_stopping()):
                break
        act=self.predict_action()
        self.step(act)
        actions[i+1]=act
        return actions[:i+2]
    def run_roll(self,roll_steps:int,early_stopping:bool=False)->np.ndarray:
        """Run the step roll."""
        self.pre_roll_initialize()
        actions=np.full((roll_steps,),self.game_env.action_nop_id,dtype=np.int16,order="C")
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
    def hook_after_step(self,action:int=-1)->None:
        """Game Hook: executed at the end of the game step."""
        return
    def hook_activated_event_flag(self,flag_name:str)->None:
        """Game Hook: executed after activating an event flag."""
        return
