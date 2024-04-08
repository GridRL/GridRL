#!/usr/bin/env python3

"""
Submodule [abstract_games] __init__.
Groups default abstracts game cores.
"""

if __package__ is None or len(__package__) == 0:
    import sys
    import os
    try:
        __file__ is None
    except NameError:
        __file__=""
    sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}{os.sep}..")
    from core_game import GameCore
    from dummy_abstract_game import DummyAbstractGame
    from exploration_abstract_game import ExplorationAbstractGame
else:
    try:
        from gridrl.core_game import GameCore
        from gridrl.abstract_games.dummy_abstract_game import DummyAbstractGame
        from gridrl.abstract_games.exploration_abstract_game import ExplorationAbstractGame
    except ModuleNotFoundError:
        from core_game import GameCore
        from abstract_games.dummy_abstract_game import DummyAbstractGame
        from abstract_games.exploration_abstract_game import ExplorationAbstractGame

__all__=[
    "GameCore",
    "DummyAbstractGame",
    "ExplorationAbstractGame",
]
