#!/usr/bin/env python3

"""Example of custom agent class."""

from typing import Any
import sys

from gridrl.agents import AgentRandomChaseWarps

sys.dont_write_bytecode = True


class AgentRandomExample(AgentRandomChaseWarps):
    """Custom subclass of an existing agent environment."""

    def __init__(self, env: Any, *args, **kwargs) -> None:
        """Constructor."""
        super().__init__(env, *args, **kwargs)
        self.dummy_attribute = 1
