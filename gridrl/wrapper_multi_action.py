#!/usr/bin/env python3

"""Wrapper to augment action space."""

from typing import Union,Any,TypeVar
import warnings
import sys
import numpy as np

sys.dont_write_bytecode=True

ObsType=TypeVar("ObsType")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from gymnasium import Env,Wrapper
    from gymnasium.spaces import MultiDiscrete,Box
    
__all__=["MultiActionWrapper","get_default_configs_for_multistep"]

class MultiActionWrapper(Wrapper):
    """Multi-action Wrapper class."""
    def __init__(self,env:Env,env_config:Union[dict,None]=None,*args,**kwargs)->None:
        """Constructor."""
        super().__init__(env)
        self.env=env
        self.disable_reward=False
        self.init_multiaction(env_config)
    def init_multiaction(self,config:dict)->None:
        """Initialize multi-action configuration."""
        self.disable_reward=False
        self.using_multistep=bool(config.get("use_multistep",False))
        self.max_directional_movements=max(1,min(5,int(config.get("action_multi_max_directional_movements",1))))
        self.angular_multistep=bool(config.get("action_multi_angular",False))
        if not self.using_multistep:
            return
        self.enable_axis_order=bool(config.get("action_multi_enable_axis_order",True))
        self.non_directional_actions_count=max(-1,min(15,int(config.get("action_multi_non_directional_actions_count",-1))))
        if self.non_directional_actions_count<0:
            self.env.non_directional_actions_count=self.env.allowed_actions_count-self.env.movement_max_actions
        self.valid_reward_methods={"regular":(lambda x:x[-1]),"only_last":(lambda x:x[-1]),"mean":np.mean,"max":np.max,"custom":np.mean}
        reward_method=config.get("action_multi_reward_method","only_last")
        if callable(reward_method):
            (self.reward_method,self.valid_reward_methods["custom"])=("custom",reward_method)
        else:
            self.reward_method=reward_method if isinstance(reward_method,str) and reward_method in self.valid_reward_methods else "only_last"
        self.original_action_space=self.action_space
        if self.angular_multistep:
            self.action_space=Box(low=-np.pi,high=np.pi,shape=[1],dtype=np.float64)
        else:
            action_space_shape=[1+self.max_directional_movements*2,1+self.max_directional_movements*2]
            if self.enable_axis_order:
                action_space_shape.append(2)
            if self.non_directional_actions_count>1:
                action_space_shape.append(self.non_directional_actions_count)
            self.action_space=MultiDiscrete(action_space_shape)
    def predict_action(self)->list:
        """Predict an action if there is an agent linked or pick one randomly otherwise."""
        if not self.using_multistep or not self.has_agent():
            return self.env.predict_action()
        actions=[]
        for _ in range(self.max_directional_movements):
            act=self.env.predict_action()
            actions.append(act)
            if act>=self.movement_max_actions or len(self.agent.scheduled_actions)==0:
                break
        return self.encode_actions(actions)
    def encode_single_directional_action(self,action:int)->tuple:
        """Convert a directional action into (axis,offsets from player) format."""
        offs=self.get_action_offset(action)
        axis=0 if offs[1]==0 else 1
        return (axis,offs[axis])
    def decode_single_directional_action(self,axis:int,off:int)->int:
        """Decode (axis,offsets) into a directional action."""
        if off==0:
            return 0
        if axis==0:
            return 0 if off>0 else 3
        return 2 if off>0 else 1
    def encode_button_action(self,button:int)->int:
        """Encode a non-directional action."""
        return button-self.movement_max_actions+1
    def decode_button_action(self,button:int)->int:
        """Decode a non-directional action."""
        return button+self.movement_max_actions-1
    def encode_angle(self,enc_y_off:int,enc_x_off:int,actions_size:int)->float:
        """Encode offsets movements into an angle."""
        return np.clip(np.arctan2(enc_y_off-self.max_directional_movements,enc_x_off-self.max_directional_movements),-np.pi,np.pi)
    def decode_angle(self,angle:float)->list:
        """Decode an angle into offsets movements."""
        offs=np.array([np.sin(angle),np.cos(angle)])*self.max_directional_movements
        return (np.round(offs/np.abs(offs).sum()*self.max_directional_movements).astype(np.int16)+self.max_directional_movements).tolist()
    def encode_actions(self,action:Union[list,np.ndarray])->list:
        """Encode a list of actions."""
        for max_rep in range(self.max_directional_movements,0,-1):
            enc_actions=[self.max_directional_movements,self.max_directional_movements]
            non_directional_idx=self.max_directional_movements
            last_axis=0
            for i,act in enumerate(action[:max_rep]):
                if act<self.movement_max_actions:
                    (last_axis,off)=self.encode_single_directional_action(act)
                    enc_actions[last_axis]+=off
                else:
                    non_directional_idx=i
                    break
            if not np.array_equal(enc_actions,[self.max_directional_movements,self.max_directional_movements]):
                break
        if self.angular_multistep:
            enc_actions=[self.encode_angle(*enc_actions[:2],max_rep)]
        else:
            if self.enable_axis_order:
                enc_actions.append(last_axis^1)
            if self.non_directional_actions_count>1:
                enc_actions.append(self.encode_button_action(action[non_directional_idx]) if non_directional_idx<self.max_directional_movements else 0)
        return enc_actions
    def decode_actions(self,action:Union[int,list,np.ndarray])->list:
        """Decode a list of actions."""
        if isinstance(action,int):
            return [action]
        dec_actions=[]
        if self.angular_multistep:
            action_2d=self.decode_angle(action[0])+(action.tolist() if isinstance(action,np.ndarray) else action)[1:]
            if len(action_2d)<3:
                action_2d.append(0)
        else:
            action_2d=action
        for axis in ([action_2d[2],action_2d[2]^1] if self.enable_axis_order else range(2)):
            off=int(action_2d[axis])-self.max_directional_movements
            for _ in range(abs(off)):
                dec_actions.append(self.decode_single_directional_action(axis,off))
        if self.non_directional_actions_count>1:
            for act in action_2d[-1:]:
                if act>0:
                    dec_actions.append(self.decode_button_action(act))
        if len(dec_actions)==0:
            dec_actions.append(self.original_action_space.sample())
        return dec_actions
    def update_reward(self,action:int=-1)->float:
        """Delta reward of the action at the current step."""
        return 0 if self.disable_reward else self.env.update_reward(action)
    def reset(self,*,seed:Union[int,None]=None,options:Union[dict[str,Any],None]=None)->tuple[ObsType,dict[str,Any]]:
        """Resets the environment to an initial internal state, returning an initial observation and info."""
        return self.env.reset(seed=seed,options=options)
    def step(self,action,return_all:bool=False)->tuple:
        """Step environment."""
        if self.using_multistep:
            return self.multi_step(action,return_all)
        rets=[self.env.step(action)]
        if self.has_agent():
            for _ in range(1,self.max_directional_movements):
                rets.append(self.env.step(self.predict_action()))
        return rets if return_all else rets[-1]
    def should_break_multistep(self)->bool:
        """Check if multi-stepping should interrupt. Required for warp consistency."""
        return self.env.game_state["pending_warp_state"]==1
    def multi_step(self,action:Union[int,list,np.ndarray],return_all:bool=False)->tuple:
        """Apply mulitple step on the environment converting the action into multiple ones."""
        rewards=[]
        dec_actions=self.decode_actions(action)
        last_action_idx=len(dec_actions)-1
        self.disable_reward=self.reward_method=="only_last"
        all_steps_data=[]
        for i,act in enumerate(dec_actions):
            if i==last_action_idx:
                self.disable_reward=False
            step_data=self.env.step(act)
            all_steps_data.append(step_data)
            if not self.disable_reward:
                rewards.append(step_data[1])
            if step_data[2] or self.should_break_multistep():
                break
        if self.disable_reward:
            self.disable_reward=False
            rewards.append(self.update_reward(dec_actions[-1]))
        if len(all_steps_data)==0:
            all_steps_data=[[self.get_observations(),self.update_reward(0),self.is_done(),self.is_done(),{}]]
        elif not return_all:
            if isinstance(all_steps_data[-1],tuple):
                all_steps_data[-1]=list(all_steps_data[-1])
            all_steps_data[-1][1]=self.valid_reward_methods[self.reward_method](rewards)
        if return_all:
            for i,v in enumerate(all_steps_data):
                all_steps_data[i]=list(v)+[dec_actions[i]]
        return all_steps_data if return_all else tuple(all_steps_data[-1])

def get_default_configs_for_multistep(kind=0,config=None):
    """Utility function for multi-step config."""
    new_config={} if config is None else dict(config)
    #kind=0
    new_config.update({
        "use_multistep":True,
        "action_multi_max_directional_movements":3,
#        "action_multi_angular":True,
#        "action_multi_enable_axis_order":True,
        "action_multi_non_directional_actions_count":0,
        "action_multi_reward_method":"only_last",
    })
    return new_config
