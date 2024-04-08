#!/usr/bin/env python3

"""
Dummy submodule [envs] __init__.
Groups default game environments.
"""

if __package__ is None or len(__package__) == 0:
    import sys
    import os
    try:
        __file__ is None
    except NameError:
        __file__=""
    sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}{os.sep}..")
    from core_environment import (deepcopy_env,ensure_env,make_env_class,make_env,
            validate_environment,GridRLAbstractEnv)
    from games.exploration_world1.env import ExplorationWorld1Env
    from games.creatures_world1.env import CreaturesWorld1Env
else:
    try:
        from gridrl.core_environment import (deepcopy_env,ensure_env,make_env_class,make_env,
            validate_environment,GridRLAbstractEnv)
        from gridrl.games.exploration_world1.env import ExplorationWorld1Env
        from gridrl.games.creatures_world1.env import CreaturesWorld1Env
    except ModuleNotFoundError:
        from .core_environment import (deepcopy_env,ensure_env,make_env_class,make_env,
            validate_environment,GridRLAbstractEnv)
        from .games.exploration_world1.env import ExplorationWorld1Env
        from .games.creatures_world1.env import CreaturesWorld1Env

__all__=[
    "deepcopy_env",
    "ensure_env",
    "make_env_class",
    "make_env",
    "validate_environment",
    "GridRLAbstractEnv",
    "ExplorationWorld1Env",
    "CreaturesWorld1Env",
]
