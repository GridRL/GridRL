#!/usr/bin/env python3

"""Example of custom environment class."""

import sys
import numpy as np

from gymnasium.spaces import Box
from gridrl.envs import CreaturesWorld1Env

sys.dont_write_bytecode = True

#__all__=[]

class ExplorationWorld1ExampleEnv(CreaturesWorld1Env):
    """Custom subclass of an existing game environment."""

    def __init__(self, config: dict = None, *args, **kwargs) -> None:
        """Constructor."""
        super().__init__(config,*args,**kwargs)
    #########################
    ### PEP COMPATIBILITY ###
    ##########################################
    ### YOU CAN SAFELY DEFINE IN hook_init ###
    ##########################################
        self.stepped_coords = {}
        self.stepped_warps = {}
        self.stepped_maps_ch0 = {}
        self.stepped_maps_ch1 = {}

    ##############################
    ### USER DEFINED FUNCTIONS ###
    ##############################
    def format_coords(self):
        """String encoding of player coordinates and map."""
        return "{:d}|{:d}|{:d}".format(*self.game_state["player_coordinates_data"][:3])

    ##################
    ### GAME HOOKS ###
    ##################
    def hook_init(self) -> None:
        """Game Hook: executed at game initialization."""
        self.stepped_coords = {}
        self.stepped_warps = {}
        self.stepped_maps_ch0 = {}
        self.stepped_maps_ch1 = {}

    def hook_reset(self) -> None:
        """Game Hook: executed at game reset."""
        self.stepped_coords = {self.format_coords(): self.step_count}
        self.stepped_warps = {}
        self.stepped_maps_ch0 = {}
        self.stepped_maps_ch1 = {}
        if self.game_state["player_coordinates_data"][4] == 0:
            self.stepped_maps_ch0[self.format_coords()] = self.step_count
        else:
            self.stepped_maps_ch1[self.format_coords()] = self.step_count

    def hook_before_warp(self, global_warped: bool, movements: list) -> None:
        """Game Hook: executed before entering a warp."""
        if not global_warped:
            self.stepped_warps[self.format_coords()] = self.step_count

    def hook_after_warp(self) -> None:
        """Game Hook: executed after exiting a warp."""
        if self.game_state["player_coordinates_data"][4] == 0:
            self.stepped_maps_ch0[self.format_coords()] = self.step_count
        else:
            self.stepped_maps_ch1[self.format_coords()] = self.step_count
            self.stepped_warps[self.format_coords()] = self.step_count

    def hook_after_movement(self) -> None:
        """Game Hook: executed after moving a tile."""
        self.stepped_coords[self.format_coords()] = self.step_count

    def hook_after_step(self, action: int = -1) -> None:
        """Game Hook: executed at the end of the game step."""
        return

    def hook_after_script(
        self, key: str, script_data: list, should_delete_script: bool = False
    ) -> None:
        """Game Hook: executed after a script runs."""
        return

    def hook_after_event(
        self, key: str, event_data: list, use_script_positions: bool = False
    ) -> None:
        """Game Hook: executed after an event is activated."""
        return

    def hook_update_overworld_screen(self) -> None:
        """Game Hook: executed updating the screen while in overworld."""
        return

    ###################
    ### DEBUG HOOKS ###
    ###################
    def hook_get_debug_text(self) -> str:
        """Extra text printed in gui mode for debug purposes."""
        return ""

    ##################
    ### DATA NAMES ###
    ##################
    def get_extra_attribute_state_names(self) -> list[str]:
        """List of extra attribute names preserved in a save state."""
        return [
            "stepped_coords",
            "stepped_warps",
            "stepped_maps_ch0",
            "stepped_maps_ch1",
        ]

    #########################
    ### GYM-ENV UTILITIES ###
    #########################
    def env_is_done(self) -> bool:
        """Return True to stop the game execution."""
        return False

    def env_reward(self, action: int = -1) -> float:
        """Total reward of the action at the current step."""
        exploration_reward = 0.005 + len(self.stepped_coords)
        events_reward = 5.0 * self.get_event_flags_sum() + 25.0 * self.game_completed
        return exploration_reward + events_reward

    def update_reward(self, action: int = -1) -> float:
        """Delta reward of the action at the current step."""
        self.check_new_event_reward_flags()
        new_total = self.env_reward(action)
        new_step = new_total - self.total_reward
        self.total_reward = new_total
        return new_step

    def get_nonscreen_observations(self) -> dict[np.ndarray]:
        """Main method to declare the step non-screen observations."""
        return {"dummy_ndarray": np.zeros((4, 4), dtype=np.float32)}

    def get_nonscreen_observation_space(self) -> dict:
        """Declarations of the get_nonscreen_observations space types."""
        return {"dummy_ndarray": Box(low=-1, high=1, shape=(4, 4), dtype=np.float32)}
