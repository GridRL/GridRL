#!/usr/bin/env python3

"""Constants for the game exploration_world1."""

from typing import Union,Any
import sys
import os
sys.dont_write_bytecode=True

if __package__ is None or len(__package__)==0:
    try:
        __file__=__file__
    except NameError:
        __file__=""
    sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}{os.sep}..{os.sep}..")
    from game import ExplorationWorld1Game
else:
    from .game import ExplorationWorld1Game
try:
    from gridrl.core_environment import GridRLAbstractEnv
except ModuleNotFoundError:
    from core_environment import GridRLAbstractEnv

__all__=["ExplorationWorld1Env"]

class ExplorationWorld1Env(ExplorationWorld1Game,GridRLAbstractEnv):
    """Environment declaration from ExplorationWorld1Game and GridRLAbstractEnv classes."""
    def __init__(self,
        config:Union[dict,None]=None,
        agent_class:Union[Any,None]=None,
        agent_args:Union[dict,None]=None,
        *args,**kwargs
    )->None:
        """Constructor to initialize Game and gymnasium Env inherited classes."""
        GridRLAbstractEnv.__init__(self,config)
        ExplorationWorld1Game.__init__(self,config=config,agent_class=agent_class,agent_args=agent_args,*args,**kwargs)
        self.post_init_binding()
        self.update_observation_spaces()
