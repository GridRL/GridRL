#!/usr/bin/env python3

"""Test of the deterministic agent implementation."""

import sys

from gridrl.envs import CreaturesWorld1Env
from example_agent import AgentRandomExample

sys.dont_write_bytecode = True

#__all__=[]

def test_agent() -> None:
    """Testing if the agent is entering a seen warp."""
    env = CreaturesWorld1Env({"rewind_steps": 2})
    env.force_teleport_to(0x26, 4, 4, 1)
    env.show_agent_screen()
    agent = AgentRandomExample(env, chase_freq=1.0)
    for _ in range(7):
        env.step(env.predict_action())
    env.show_agent_screen()
    assert env.game_state["player_coordinates_data"][0] == 0x25
    env.rewind_state(9)
    env.show_agent_screen()
    assert env.game_state["player_coordinates_data"][0] == 0x26
    for _ in range(7):
        ### THIS DOESN'T RETURNING OBSERVATIONS AND IS FASTER
        env.step_game_predict()
    env.rewind_state(9)
    ### EQUIVALENT WITHOUT OBSERVATIONS
    agent.run_roll(7)
    env.show_agent_screen()


if __name__ == "__main__":
    test_agent()
