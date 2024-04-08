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
    from agents_base import AgentBase
    from agents_random import AgentRandom,AgentRandomOffsets,AgentRandomChaseWarps
else:
    try:
        from gridrl.agents_base import AgentBase
        from gridrl.agents_random import AgentRandom,AgentRandomOffsets,AgentRandomChaseWarps
    except ModuleNotFoundError:
        from .agents_base import AgentBase
        from .agents_random import AgentRandom,AgentRandomOffsets,AgentRandomChaseWarps


__all__=[
    "AgentBase",
    "AgentRandom",
    "AgentRandomOffsets",
    "AgentRandomChaseWarps",
]
