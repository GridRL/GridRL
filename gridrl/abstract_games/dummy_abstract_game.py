#!/usr/bin/env python3

"""Declaration of dummy_abstract_game."""

from typing import Union,Any
import sys
import os
sys.dont_write_bytecode=True

if __package__ is None or len(__package__)==0:
    try:
        __file__=__file__
    except NameError:
        __file__=""
    sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}{os.sep}..")
    from core_game import GameCore
else:
    try:
        from gridrl.core_game import GameCore
    except ModuleNotFoundError:
        from core_game import GameCore

__all__=["DummyAbstractGame"]

class DummyAbstractGame(GameCore):
    """The main implementation of dummy_abstract_game."""
    def __init__(self,
        game_name:str,
        config:Union[dict,None]=None,
        agent_class:Union[Any,None]=None,
        agent_args:Union[dict,None]=None,
        *args,**kwargs
    )->None:
        """Constructor."""
        super().__init__(game_name=game_name,config=config,
            agent_class=agent_class,agent_args=agent_args,*args,**kwargs)
