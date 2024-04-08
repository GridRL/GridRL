#!/usr/bin/env python3

"""Basic agents implementations."""

import sys
import numpy as np
sys.dont_write_bytecode=True

if __package__ is None or len(__package__)==0:
#    from functions_numba import nb_is_point_inside_area
    from core_environment import GridRLAbstractEnv
    from agents_base import AgentBase
else:
#    from gridrl.functions_numba import nb_is_point_inside_area
    from gridrl.core_environment import GridRLAbstractEnv
    from gridrl.agents_base import AgentBase

def normalize_sum_one(nums:np.ndarray)->np.ndarray:
    """Normalize ndarray to sum to 1 on last axis."""
    return np.divide(nums,np.expand_dims(np.sum(nums,axis=-1),axis=-1))

class AgentRandom(AgentBase):
    """Random Agent class."""
    def __init__(self,env:GridRLAbstractEnv,sticky_freq:float=0.,*args,**kwargs)->None:
        """Constructor."""
        super().__init__(env,*args,**kwargs)
        self.sticky_freq=min(0.,max(0.99,float(sticky_freq)))
    def run_roll(self,roll_steps:int,early_stopping:bool=False)->np.ndarray:
        """Run the step roll."""
        self.pre_roll_initialize()
        actions=np.full((roll_steps,),self.env.action_nop_id,dtype=np.int16,order="C")
        prev_act=0
        i=0
        for i,(act,stick) in enumerate(zip(self.env.roll_random_actions_without_nop(roll_steps),
            np.ones((roll_steps,),dtype=np.float32,order="C") if self.sticky_freq<1e-6 else np.random.rand(roll_steps))):
            if stick<self.sticky_freq and act<self.env.movement_max_actions:
                act=prev_act if self.env.movement_max_actions==4 else 0
            data=self.env.step(act)
            actions[i]=act
            if data[2] or (early_stopping and self.check_early_stopping()):
                break
            prev_act=act
        return actions[:i+1]

class AgentRandomOffsets(AgentBase):
    """Random-Offset Agent class."""
    def __init__(self,env:GridRLAbstractEnv,sticky_freq:float=0.,*args,**kwargs)->None:
        """Constructor."""
        super().__init__(env,sticky_freq,*args,**kwargs)
        self.axis_priority=0
    def predict_action(self)->int:
        """Get the next predicted action."""
        if len(self.scheduled_actions)>0:
            return self.pop_last_scheduled_action()
        for i in range(64):
            target=self.env.roll_random_screen_offsets(1)[0]
            coords=self.env.game_state["player_coordinates_data"][1:3].copy()+target
            sizes=self.env.get_cached_map_sizes(self.env.game_state["player_coordinates_data"][0])
            if sizes[0]>coords[0]>=0 and sizes[1]>coords[1]>=0 and not np.array_equal(target,[0,0]):
#            if nb_is_point_inside_area(*coords,0,0,*self.env.get_cached_map_sizes(self.env.game_state["player_coordinates_data"][0])) and not np.array_equal(target,[0,0]):
                break
        offs=np.clip(target,-1,1)
        max_travel=np.abs(target).clip(0,np.abs(self.env.player_screen_bounds).max())
        actions=[]
        self.axis_priority^=1
        for i in [0,1][slice(None,None,None if self.axis_priority==0 else -1)]:
            toffs=offs.copy()
            toffs[i^1]=0
            actions+=[self.env.get_action_from_direction_offsets_4way(*toffs) for i in range(np.abs(max_travel[i]))]
        #if np.random.shuffle(actions)
### TO-DO: ADD FINAL DIRECTION FOR CONSISTENCY WITH 2-3 WAY INPUT DIRECTIONS
        self.set_scheduled_actions(actions)
        return self.pop_last_scheduled_action()

class AgentRandomChaseWarps(AgentBase):
    """Random warp-chaser Agent class."""
    def __init__(self,env:GridRLAbstractEnv,sticky_freq:float=0.,chase_freq:float=0.25,*args,**kwargs)->None:
        """Constructor."""
        super().__init__(env,sticky_freq,*args,**kwargs)
        self.chase_freq=chase_freq
    def predict_action(self)->int:
        """Get the next predicted action."""
        if len(self.scheduled_actions)>0:
            return self.pop_last_scheduled_action()
        if np.random.rand()<self.chase_freq:
            map_warps=self.env.get_cached_map_warps(self.env.game_state["player_coordinates_data"][0])
            if len(map_warps)>0:
                warp_offs=np.array([w[:2]-self.env.game_state["player_coordinates_data"][1:3] for w in map_warps])
                distances_metric=1+np.power((np.sum(np.abs(warp_offs),axis=1)-1).clip(0),2)
                distances_metric[distances_metric>64]=0xFFF
                probs=normalize_sum_one(1./distances_metric)
                warp_index=np.random.choice(np.arange(len(probs),dtype=np.uint8),p=probs,replace=False,size=1)[0]
                offs=np.clip(warp_offs[warp_index],-1,1)
                max_travel=np.abs(warp_offs[warp_index]).clip(0,5)
                actions=[]
                for i in range(2):
                    toffs=offs.copy()
                    toffs[i^1]=0
                    actions+=[self.env.get_action_from_direction_offsets_4way(*toffs) for i in range(np.abs(max_travel[i]))]
                np.random.shuffle(actions)
                self.set_scheduled_actions(actions)
                if len(self.scheduled_actions)>0:
                    return self.pop_last_scheduled_action()
        return super().predict_action()
