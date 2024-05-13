#!/usr/bin/env python3

"""Base gymnasium environment of any GameCore."""

from typing import Union,Callable,Any,SupportsFloat,TypeVar
from collections import defaultdict
import warnings
import sys
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import numpy as np
    import pandas as pd
    import gymnasium
    from gymnasium import Env
    from gymnasium.spaces import Discrete,MultiDiscrete,MultiBinary,Box,Dict

sys.dont_write_bytecode=True
sys.modules["gym"]=gymnasium


if __package__ is None or len(__package__)==0:
    from graph import Graph
    from games_list import GAMES_LIST
    from functions_dataframe import is_dataframe,read_dataframe,write_dataframe
    from game_module_selector import get_game_default_env_class
    from core_game import GameAbstractCore
    from wrapper_gui import GuiWrapper
    from wrapper_multi_action import MultiActionWrapper
    from wrapper_stream_agent import StreamWrapper
else:
    from gridrl.graph import Graph
    from gridrl.games_list import GAMES_LIST
    from gridrl.functions_dataframe import is_dataframe,read_dataframe,write_dataframe
    from gridrl.game_module_selector import get_game_default_env_class
    from gridrl.core_game import GameAbstractCore
    from gridrl.wrapper_gui import GuiWrapper
    from gridrl.wrapper_multi_action import MultiActionWrapper
    from gridrl.wrapper_stream_agent import StreamWrapper

ObsType=TypeVar("ObsType")
ActType=TypeVar("ActType")
RenderFrame=TypeVar("RenderFrame")

def warn(*args,**kwargs)->None:
    """Suppressing warnings."""
    return
warnings.simplefilter(action="ignore",category=FutureWarning)
warnings.warn=warn
def  no_warning()->None:
    """Suppressing warnings."""
    warnings.warn("deprecated",DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    no_warning()

__all__=["seed_everything","GridRLAbstractEnv","deepcopy_env","ensure_env",
    "make_env_class","make_env","validate_environment"]

def seed_everything(seed)->None:
    return

class GridRLAbstractEnv(Env,GameAbstractCore):
    """Abstract GameCore gymnasium class for implementing Reinforcement Learning Agents environments."""
    def __init__(self,config,*args,**kwargs)->None:
        """Constructor."""
        Env.__init__(self)
        GameAbstractCore.__init__(self)
        self.env_initialized=False
        self.flat_obs=bool(config.get("flat_obs",False))
        self.auto_screen_obs=bool(config.get("auto_screen_obs",True))
        try:
            self.flat_obs_dtype=np.dtype(config.get("flat_obs_dtype",None)).type
        except TypeError:
            self.flat_obs_dtype=None
        self.set_env_id(config.get("env_id",0))
        self.metadata["render.modes"]=[]
        self.screen_box_high=1
        self.screen_box_shape=[1,1]
        self.action_space=None
        self.screen_observation_space=None
        self.observation_space=None
        self.update_observation_spaces()
        self.pre_init_binding()
###########################
### ENVIRONMENT METHODS ###
###########################
    def pre_init_binding(self)->None:
        """Binds the game data with the gym environment, before the game initialization."""
    def post_init_binding(self)->None:
        """Binds the game data with the gym environment, after the game initialization."""
        return
    def update_observation_spaces(self)->bool:
        """Updates the observation spaces depending on game and configs."""
        self.action_space=Discrete(self.all_actions_count)
        raw_obs_space_dict={}
        if self.auto_screen_obs:
            self.screen_observation_space=Box(low=0,high=self.screen_box_high,shape=self.screen_box_shape,dtype=np.uint8)
            raw_obs_space_dict["screen"]=self.screen_observation_space
        else:
            self.screen_observation_space=None
        try:
            raw_obs_space_dict.update(self.get_nonscreen_observation_space())
        except AttributeError:
            return False
        self.unflattened_observation_space=Dict(raw_obs_space_dict)
        try:
            test_obs=self.get_observations()
        except AttributeError:
            return False
        (max_bits,use_signed,use_float)=([8],False,False)
        for k,v in test_obs.items():
            if k not in raw_obs_space_dict:
                raw_obs_space_dict[k]=Box(low=-127,high=128,shape=v.shape,dtype=v.dtype)
            if isinstance(raw_obs_space_dict[k],(Discrete,MultiDiscrete,MultiBinary)):
                continue
            elif hasattr(raw_obs_space_dict[k],"dtype"):
                try:
                    dti=np.iinfo(raw_obs_space_dict[k].dtype)
                    vmax=int(8*np.round(np.log2(dti.max+1)/8.))
                    if dti.min<0:
                        use_signed=True
                except ValueError:
                    dti=np.finfo(raw_obs_space_dict[k].dtype)
                    vmax=int(np.finfo(raw_obs_space_dict[k].dtype).bits)
                    use_signed=True
                    use_float=True
                max_bits.append(vmax)
        self.flat_obs_dtype=np.dtype(f"{'float' if use_float else ('int' if use_signed else 'uint')}{np.max(max_bits):d}")
        try:
            test_obs=self.flatten_observations(test_obs)
            dtype=test_obs.dtype if self.flat_obs_dtype is None else self.flat_obs_dtype
            try:
                obs_dti=np.iinfo(dtype)
            except ValueError:
                obs_dti=np.finfo(dtype)
            (flatten_dim,low,high)=(len(test_obs),obs_dti.min,obs_dti.max)
        except (AttributeError,ValueError,IndexError,TypeError,NameError):
            (flatten_dim,low,high,dtype)=(0,0,255,np.int32 if self.flat_obs_dtype is None else self.flat_obs_dtype)
            for _,v in raw_obs_space_dict.items():
                if isinstance(v,Box):
                    flatten_dim+=np.prod(v.shape)
        self.flattened_observation_space=Box(low=low,high=high,shape=[flatten_dim],dtype=dtype)
        self.flat_obs_dtype=dtype
        self.observation_space=self.flattened_observation_space if self.flat_obs else self.unflattened_observation_space
        if not self.env_initialized:
            Env.__init__(self)
            self.env_initialized=True
        return True
    def set_env_id(self,env_id:int)->None:
        """Set environment id for streaming purposes."""
        self.env_id=max(0,int(env_id))
    def flatten_observations(self,obs:Union[dict,np.ndarray])->np.ndarray:
        """Flatten the observations to a 1d ndarray."""
        obs=np.hstack([obs[k].flatten() if isinstance(obs[k],np.ndarray) else np.expand_dims(obs[k],axis=0) for k,_ in self.unflattened_observation_space.items()],dtype=self.flat_obs_dtype) if isinstance(obs,dict) else obs.flatten()
        if self.flat_obs_dtype is not None and obs.dtype!=self.flat_obs_dtype:
            obs=obs.astype(self.flat_obs_dtype)
        return obs
        #return obs if self.flat_obs_dtype is None or obs.dtype==self.flat_obs_dtype else obs.astype(self.flat_obs_dtype)
    def unflatten_observations(self,obs:Union[dict,np.ndarray],expand_flat:bool=True,downcast:bool=True)->dict:
        """Unflatten the observation to dict of ndarray with shape [batch,*obs_shape]."""
        if isinstance(obs,dict):
            return obs
        unflattened={}
        base_shape=list(obs.shape[:-1])
        flat_pos=0
        for k,s in self.unflattened_observation_space.items():
            cur_shape=list(base_shape)
            if hasattr(s,"shape") and (not expand_flat or len(s.shape)>0):
                cur_shape.extend(s.shape)
            else:
                cur_shape.append(1)
            cur_size=max(1,np.prod(s.shape))
            unflattened[k]=obs[...,flat_pos:flat_pos+cur_size]
            if len(cur_shape)>0:
                unflattened[k]=unflattened[k].reshape(cur_shape)
            if hasattr(s,"dtype"):
                if unflattened[k].dtype!=s.dtype:
                    unflattened[k]=unflattened[k].astype(s.dtype)
            elif downcast:
                pass#
            if isinstance(unflattened[k],np.ndarray) and not unflattened[k].flags["C_CONTIGUOUS"]:
                unflattened[k]=np.ascontiguousarray(unflattened[k])
            flat_pos+=cur_size
        return unflattened
    def step(self,action:ActType)->tuple[ObsType,SupportsFloat,bool,bool,dict[str,Any]]:
        """Run one timestep of the environment's dynamics using the agent actions."""
        self.step_game(action)
        return self.get_step_return_tuple(action)
    def reset(self,*,seed:Union[int,None]=None,options:Union[dict[str,Any],None]=None)->tuple[ObsType,dict[str,Any]]:
        """Resets the environment to an initial internal state, returning an initial observation and info."""
        if seed is not None and seed>=0:
            super().reset(seed=seed,options=options)
        self.reset_game(seed)
        if seed is not None:
            try:
                seed_everything(seed)
            except NameError:
                pass
        return (self.flatten_observations(self.get_observations())
            if self.flat_obs else self.get_observations(),{})
    def is_truncate(self)->bool:
        """Return the truncate environment flag."""
        return False
    def render(self)->RenderFrame|list[RenderFrame|None]:
        """Compute the screen rendering."""
        return
    def get_step_return_tuple(self,action:int)->tuple:
        """Build the environment step return tuple."""
#        return
        new_reward=self.update_reward(action)
        return (self.flatten_observations(self.get_observations())
                if self.flat_obs else self.get_observations(),
            new_reward,self.is_done(),self.is_truncate(),{})
    def step_flatten_obs(self,action:int)->tuple:
        """Run the environment step forcing flatted observations."""
        if self.flat_obs:
            return self.step(action)
        (obs,reward,done,truncated,info)=self.step(action)
        return (self.flatten_observations(obs),reward,done,truncated,info)
    def run(self,steps:int,stop_done:bool=True,*args,**kwargs)->None:
        """Run the environment for N steps using the agent predictions."""
        if steps<1:
            while True:
                (_,_,done,_,_)=self.step(self.predict_action())
                if done and stop_done:
                    break
        else:
            for _ in range(steps):
                (_,_,done,_,_)=self.step(self.predict_action())
                if done and stop_done:
                    break
    def run_quit(self,steps:int=-1,stop_done:bool=True,**kwargs)->None:
        """Run the environment, saves a gif and quits execution."""
        self.run(steps,stop_done)
        try:
            self.save_run_gif()
        except AttributeError:
            pass
        try:
            self.close()
        except AttributeError:
            pass
        sys.exit(1)
#######################
### DATASET METHODS ###
#######################
    def run_ml_dataset(self,steps:int=16384,discard_uncompleted:bool=False):
        """Run the environment for multiple steps and return stacked data in dataset format."""
        (stack_obs,actions)=([],[])
        for i in range(steps):
            act=self.predict_action()
            actions.append(act)
            (obs,reward,done,truncated,info)=self.step_flatten_obs(act)
            stack_obs.append(obs)
            if done:
                break
        if len(stack_obs)==0 or (discard_uncompleted and not self.game_completed):
            return (np.zeros((0,1),dtype=np.uint8,order="C"),np.zeros((0,),dtype=np.uint8,order="C"))
        stack_obs=np.vstack(stack_obs,dtype=stack_obs[0].dtype)
        actions=np.hstack(actions).astype(np.uint8)
        return (stack_obs,actions)
    def run_multiple_ml_dataset(self,runs:int,steps:int=16384,discard_uncompleted:bool=False):
        """Run multiple indipendent runs and return data in dataset format."""
        (stack_obs,actions,run_sizes)=([],[],[])
        for rep in range(runs):
            self.reset()
            (cur_stack_obs,cur_actions)=self.run_ml_dataset(steps,discard_uncompleted)
            if len(cur_actions)>0:
                stack_obs.append(cur_stack_obs)
                actions.append(cur_actions)
                run_sizes.append(len(cur_actions))
        if len(stack_obs)==0:
            return (np.zeros((0,1),dtype=np.uint8,order="C"),
                np.zeros((0,),dtype=np.uint8,order="C"),
                np.zeros((0,),dtype=np.uint32,order="C"))
        stack_obs=np.vstack(stack_obs,dtype=stack_obs[0].dtype)
        actions=np.hstack(actions,dtype=np.uint8)
        run_sizes=np.hstack(run_sizes).astype(dtype=np.uint32)
        return (stack_obs,actions,run_sizes)
    def get_obs_dataframe_columns_and_dtypes(self,default_dtype:Union[np.dtype,None]=None)->tuple[np.ndarray,np.ndarray]:
        """Get observation column names and dtypes."""
        (columns,dtypes)=([],[])
        for k,s in self.unflattened_observation_space.items():
            cur_shape=[]
            if hasattr(s,"shape") and len(s.shape)>0:
                cur_shape.extend(s.shape)
            else:
                cur_shape.append(1)
            cur_size=max(1,np.prod(s.shape))
            dtypes.extend([s.dtype if hasattr(s,"dtype") else (self.flat_obs_dtype if default_dtype is None else default_dtype)]*cur_size)
            for i in range(cur_size):
                columns.append(f"obs|{k}_f{i:d}")
        return (columns,dtypes)
    def convert_ml_dataset_to_dataframe(self,stack_obs:np.ndarray,actions:np.ndarray,
        run_sizes:np.ndarray,df_process_func:Union[Callable,None]=None
    )->pd.DataFrame:
        """Convert from dataset format to dataframe."""
        (columns,dtypes)=self.get_obs_dataframe_columns_and_dtypes(stack_obs.dtype)
        df=pd.DataFrame(stack_obs,columns=columns,dtype=stack_obs.dtype)
        #for col,dtype in zip(df.columns,dtypes):
        #    df[col]=df[col].astype(dtype)
        df["info|step_id"]=np.hstack([np.arange(k,dtype=np.uint32) for k in run_sizes])
        df["info|run_id"]=((df["info|step_id"]==0).cumsum()-1).clip(0,None)
        df["agent|action"]=actions
        df["env|reward"]=0.
        if callable(df_process_func):
            df=df_process_func(df)
        return df
    def build_and_save_runs_dataframe(self,filename:str,runs:int,steps:int=16384,
        discard_uncompleted:bool=False,df_process_func:Union[Callable,None]=None
    )->pd.DataFrame:
        """Run multiple indipendent runs and save the result in dataframe format."""
        df=self.convert_ml_dataset_to_dataframe(*self.run_multiple_ml_dataset(runs,steps,discard_uncompleted),df_process_func)
        write_dataframe(df,filename)
#        for k in df.columns:print(k)
        #print(df[["agent|action","info|run_id","info|step_id"]].to_string())
        #for k,v in stack_obs.items():
        #    print(k,v.shape)
        return df
    def load_dataframe(self,filename:str)->pd.DataFrame:
        """Load a generic dataframe from file."""
        return read_dataframe(filename)
    def load_dataset(self,filename_or_df:Union[str,pd.DataFrame],unflatten:bool=True
    )->tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Load an environment result dataframe from file."""
        df=filename_or_df if is_dataframe(filename_or_df) else self.load_dataframe(filename_or_df)
        all_columns=set(df.columns)
        (obs_columns,obs_dtypes)=self.get_obs_dataframe_columns_and_dtypes()
        obs_columns=[k for k in obs_columns if k in df.columns]
        all_columns-=set(obs_columns)
        all_columns-=set(["info|run_id","info|step_id","agent|action"])
        all_columns=sorted(all_columns)
        extra_obs_columns=[k for k in all_columns if k.endswith("_f0")]
        stack_obs=df[obs_columns].values
        actions=df["agent|action"].values
        run_sizes=df.groupby("info|run_id")["info|step_id"].apply(np.max)+1
        if unflatten:
            stack_obs=self.unflatten_observations(stack_obs)
        if not unflatten and len(extra_obs_columns)>0:
            extra_obs={}
            for k in extra_obs_columns:
                pass
                #extra_obs[k]=df[all_columns].values
        else:
            extra_obs=df[all_columns].values
        del df
        return (stack_obs,actions,run_sizes,extra_obs)

def deepcopy_env(env:GridRLAbstractEnv)->GridRLAbstractEnv:
    """Deepcopy of the environment, avoiding pickle fails due to module objects."""
    return env.deepcopy()

def ensure_env(env:Union[str,callable,GridRLAbstractEnv],
    copy:bool=False,*args,**kwargs
)->GridRLAbstractEnv:
    """Utility function to ensure an environment regardless the input type."""
    if isinstance(env,str):
        return get_game_default_env_class(env)(*args,**kwargs)
    if isinstance(env,int):
        try:
            if len(GAMES_LIST)>env>=0:
                env=GAMES_LIST[env]
        except (ValueError,TypeError):
            env=""
        return ensure_env(env,copy,*args,**kwargs)
    if env.__class__.__name__=="type":
        return env(*args,**kwargs)
    if copy:
        return deepcopy_env(env)
    return env

def make_env_class(game_env_class:Callable)->GridRLAbstractEnv:
    """Agnostic class builder to subclass a game. Placeholder for default envs."""
    class GridRLFuncMadeEnv(game_env_class,GridRLAbstractEnv):
        """Example of environment declaration from GameCore and GridRLAbstractEnv classes."""
        def __init__(self,config:Union[dict,None]=None,
            agent_class:Union[Any,None]=None,agent_args:Union[dict,None]=None,
            *args,**kwargs
        )->None:
            """Constructor to initialize GameCore and gymnasium Env inherited classes."""
            GridRLAbstractEnv.__init__(self,config)
            game_env_class.__init__(self,game_name=None,config=config,
                agent_class=agent_class,agent_args=agent_args,*args,**kwargs)
            self.post_init_binding()
            self.update_observation_spaces()
    return GridRLFuncMadeEnv

def make_env(env_class:GridRLAbstractEnv,rank:int=0,env_config:Union[dict,None]=None,
    agent_class:Union[Any,None]=None,agent_args:Union[dict,None]=None,seed:int=0,
    *args,**kwargs
)->GridRLAbstractEnv:
    """Environment builder."""
    if env_config is None:
        env_config={}
    def _init()->GridRLAbstractEnv:
        config=dict(env_config)
        config["env_id"]=rank
        multistep_keys=["use_multistep","action_multi_max_directional_movements",
            "action_multi_angular","action_multi_enable_axis_order",
            "action_multi_non_directional_actions_count","action_multi_reward_method"]
        temp_multistep_data={k:int(config[k]) if k.endswith("_id") else config[k] for k in multistep_keys if len(str(config.get(k,"")))>0}
        if len(temp_multistep_data)>1:
            config["action_nop"]=True
        env=ensure_env(env_class,copy=False,config=config)
        from_gui=bool(config.get("gui",0))>0
        if not from_gui:
            if agent_class is not None and hasattr(env,"set_agent") and callable(env.set_agent):
                env.set_agent(agent_class(env,**agent_args))
        env.reset(seed=seed+rank)
        if not from_gui and len(temp_multistep_data)>1:
            env=MultiActionWrapper(env,temp_multistep_data)
        metadata_keys=["local_stream","stream","user","env_id","sprite_id","color","extra",
            "stream_url","upload_interval"]
        temp_stream_metadata={k:v for k,v in config.get("stream_metadata",{}).items() if k in metadata_keys}
        temp_stream_metadata.update({k:int(config[k]) if k.endswith("_id") else config[k]
            for k in metadata_keys if len(str(config.get(k,"")))>0})
        if (len(temp_stream_metadata)>2 or "stream" in temp_stream_metadata
            or "local_stream" in temp_stream_metadata
        ):
            used_stream_metadata={"user":"Guest","env_id":rank,"sprite_id":0,
                "color":"#000000","extra":"","upload_interval":250,"stream_url":None}
            used_stream_metadata.update(temp_stream_metadata)
            stream_url=used_stream_metadata["stream_url"]
            upload_interval=used_stream_metadata["upload_interval"]
            for k in ["stream","stream_url","upload_interval"]:
                if k in used_stream_metadata:
                    del used_stream_metadata[k]
            env=StreamWrapper(env,stream_metadata=used_stream_metadata,
                stream_url=stream_url,upload_interval=upload_interval)
        if from_gui:
            env=GuiWrapper(env,env_config,
                agent_class=agent_class,agent_args=agent_args,play_agent=agent_class is not None,
                redirect_stdout=config.get("redirect_stdout",True))
        return env
    try:
        seed_everything(seed)
    except NameError:
        pass
    return _init

def validate_environment(env:GridRLAbstractEnv,env_config:Union[dict,None]=None,fast:bool=True,verbose:bool=True)->bool:
    """Validate environment for logical or speed inconsistencies."""
    if env_config is None:
        env_config={}
    ret=True
    try:
        env=ensure_env(env,copy=True,config=env_config)
        ret=env.validation_passed
    except (AttributeError,ValueError,IndexError,KeyError,KeyError,NameError,
        TypeError,FileNotFoundError,OSError
    ) as exc:
        ret=False
        if verbose:
            print(f"\tEncountered an Exception during environment initialization [{exc}].")
        return False
    initial_validation_passed=ret
    if not ret and not verbose:
        return False
    if verbose:
        print(f"Validating environment [{env.__class__.__name__}] of game [{env.game_selector.game_name}].")
#    try:
#        ret=env.is_valid_game_data(raise_exception=False,verbose=verbose)
#    except IndexError:
#        ret=False
    function_names=["script_core_set_automatic_map_ids",
        "script_core_set_extra_event_names","script_core_reset",
        "script_core_load_custom_save_from_starting_event",
        "script_core_automatic_with_npc","script_core_automatic_without_npc",
        "script_npc_text"]
    required_game_data_names=["channels","sizes","legacy_global_starting_coords","bounds",
        "connections_mask","warps_count","warps","maps",
        "teleport_data","events","scripts","npcs","powerups"]
    optional_game_data_names=["maps_names","global_starting_coords"]
    starting_events=[]
    graph_dict=defaultdict(dict)
    events=[]
    try:
        env.benchmarking=True
        events=list(env.event_rewards_data.keys())
        for i,(event_name,v) in enumerate(env.event_rewards_data.items()):
            if len(v[3])==0:
                starting_events.append(event_name)
            for required_event_name in v[3]:
                graph_dict[required_event_name][event_name]=i+1
        if len(events)>0:
            if len(starting_events)==0:
                starting_events.append(events[0])
            graph=Graph(graph_dict)
            for start_event in starting_events:
                for end_event in events:
                    if start_event==end_event:
                        continue
                    events_path=graph.find(start_event,end_event)
                    if len(events_path)<2:
                        if verbose:
                            ret=False
                            print(f"\tEvents path [{start_event}] -> [{end_event}] is unreachable.")
                        else:
                            return False
        else:
            if verbose:
                ret=False
                print("\tNo events set.")
            else:
                return False
        (obs,_)=env.reset()
        for dict_key in required_game_data_names:
            if dict_key not in env.game_data:
                if verbose:
                    ret=False
                    print(f"\tMissing GAME-DATA dict key [{dict_key}].")
                else:
                    return False
        for dict_key in optional_game_data_names:
            if dict_key not in env.game_data:
                if verbose:
                    print(f"\tMissing <optional> GAME-DATA dict key [{dict_key}].")
        env.validator_reinitialize()
        for func_name in function_names:
            if not env.has_direct_script(func_name):
                if verbose:
                    ret=False
                    print(f"\tMissing CORE-SCRIPT function [{func_name}].")
                else:
                    return False
        for script_name,v in env.scripts_data.items():
            if not env.has_direct_script(v[1]):
                if verbose:
                    ret=False
                    print(f"\tMissing HIDDEN [{script_name}] SCRIPT function [{v[1]}] .")
                else:
                    return False
        if env.using_npcs:
            for map_id in range(0x100):
                for npc_name,v in env.npcs_data_by_map.get(map_id,{}).items():
                    script_func_name=v[2][0] if isinstance(v[2],(list,set,tuple)) else v[2]
                    if not env.has_direct_script(script_func_name):
                        if verbose:
                            ret=False
                            print(f"\tMissing NPC [{npc_name}] SCRIPT function [{script_func_name}].")
                        else:
                            return False
        else:
            if verbose:
                ret=False
                print("\tCouldn't set NPC environment mode.")
            else:
                return False
        for attr_name in dir(env):
            v=getattr(env,attr_name)
            if isinstance(v,np.ndarray) and not v.flags["C_CONTIGUOUS"]:
                if verbose:
                    ret=False
                    print(f"\tENV-attribute [{attr_name}] ndarray not C_CONTIGUOUS.")
                else:
                    return False
        if not fast and initial_validation_passed:
            for i,act in enumerate(env.roll_random_actions_without_nop(500)):
                contiguous=True
                if isinstance(obs,np.ndarray):
                    if not obs.flags["C_CONTIGUOUS"]:
                        if verbose:
                            contiguous=False
                            print("\tENV-observation ndarray not C_CONTIGUOUS.")
                        else:
                            return False
                else:
                    for attr_name,v in obs.items():
                        if isinstance(v,np.ndarray) and not v.flags["C_CONTIGUOUS"]:
                            if verbose:
                                contiguous=False
                                print(f"\tENV-observation[{attr_name}] ndarray not C_CONTIGUOUS.")
                            else:
                                return False
                for attr_name,v in env.game_state.items():
                    if isinstance(v,np.ndarray) and not v.flags["C_CONTIGUOUS"]:
                        if verbose:
                            contiguous=False
                            print(f"\tENV-game_state[{attr_name}] ndarray not C_CONTIGUOUS.")
                        else:
                            return False
                if not contiguous:
                    ret=False
                    break
                (obs,_,done,_,_)=env.step(act)
                if done:
                    (obs,_)=env.reset()
            for attr_name in dir(env):
                v=getattr(env,attr_name)
                if isinstance(v,np.ndarray) and not v.flags["C_CONTIGUOUS"]:
                    if verbose:
                        ret=False
                        print(f"\tENV-attribute [{attr_name}] ndarray not C_CONTIGUOUS after N steps.")
                    else:
                        return False
    except (AttributeError,ValueError,IndexError,KeyError,KeyError,NameError,
        TypeError,FileNotFoundError,OSError
    ) as exc:
        if verbose:
            ret=False
            print(f"\tEncountered an Exception during validation [{exc}].")
        else:
            return False
    if ret and initial_validation_passed:
        try:
            ret=env.is_valid_game_data(raise_exception=False,verbose=verbose)
        except IndexError:
            ret=False
        if not ret and not verbose:
            return False
    if verbose and ret:
        print("\tEnvironment validation passed!")
    return ret

if __name__=="__main__":
    validate_environment(GAMES_LIST[0])
