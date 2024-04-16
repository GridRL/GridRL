#!/usr/bin/env python3

"""Environment for the game creatures_world1."""

import sys
import os
sys.dont_write_bytecode=True

if __package__ is None or len(__package__)==0:
    try:
        __file__=__file__
    except NameError:
        __file__=""
    sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}{os.sep}..{os.sep}..")
    from game import CreaturesWorld1Game
else:
    from .game import CreaturesWorld1Game
try:
    from gridrl.core_environment import GridRLAbstractEnv
except ModuleNotFoundError:
    from core_environment import GridRLAbstractEnv

__all__=["CreaturesWorld1Env"]

class CreaturesWorld1Env(CreaturesWorld1Game,GridRLAbstractEnv):
    """Environment declaration from CreaturesWorld1Game and GridRLAbstractEnv classes."""
    def __init__(self,config:dict={},agent_class=None,agent_args:dict={},*args,**kwargs)->None:
        """Constructor to initialize Game and gymnasium Env inherited classes."""
        GridRLAbstractEnv.__init__(self,config)
        CreaturesWorld1Game.__init__(self,config=config,agent_class=agent_class,agent_args=agent_args,*args,**kwargs)
        self.post_init_binding()
        self.update_observation_spaces()
