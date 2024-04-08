# GridRL
Customizable engine for minimalist 2D grid-based games, oriented towards Reinforcement Learning.
The project provides gymnasium-compatible environments with convenient hooks, of exploration games with a complexity similar to NES/GB games.
Environment state is small and contains only relevant data, so this is a good playground for training and testing policies at high speed, even on weaker devices.


Getting Started
===============
The script is currently tested on Python 3.10, more versions will be checked and supported in the future.
Minimal experience of gymnasium environment structure and Reinforcement Learning is required.
An examples using Stable-Baselines3 PPO algorithm is currently provided. Pufferlib-CleanRL implementation in project.

Install GridRL as:
```sh
$ pip install git+https://github.com/GridRL/GridRL
```

Import of the relevant environment generation functions:
```python
from gridrl.envs import make_env,ExplorationWorld1Env

config={}
env=make_env(ExplorationWorld1Env,config)
```

The CLI version is suited for benchmarks, info and debug play. It can be called on CLI or from script:
```sh
gridrl exploration_world1 --mode info
```

```sh
gridrl creatures_world1 --mode benchmark
```

```python
from gridrl.cli import main_cli

main_cli(["exploration_world1","--mode","play"])
```

Game string can also fallback to game_id and you extract example scripts in your current working directory using cli:
```sh
gridrl 0 --mode examples
```


Games structure
===============
Games follow a minimalist minimalist 2D grid-based approach, where the agent moves inside multiple maps and must navigate up to (unknown) checkpoints, that allow it to eventually modify other NPC states or unlock new powers, to be able to access new areas.
Multiple small maps are provided and assembled into a bigger global world, that can also include warps. Relevent events required for any progression of the game are given a dedicated flag, so it's possible to easily track the agent progress or start at custom points.
A hook system provides relevant bindings to the coder, that can implement custom calls before/after something occurs, like stepping on a warp.
The action space and game complexity can be partially altered by configurations, changing the way the agent moves, input/menu abstraction or filler events amount. Some games can override certain fields.
Some configurations are not yet coded and must be done on a per-game basis.
The hidden game state is as small as possible and most of the work needed for a smarter/faster policy, will be done by feature and reward enginering on spatial data that the agent can collect.


Environment configurations
==========================
Environment customization isn't documented for now, but it's not hard to understand. Check examples/example_environment.py
The configuration dictionary is optional, but suggested for whatever editing you desire. There relevant ones are:

```python

config=dict(
### Most important fields
## Size of directional action space. Default: 4
## 4:Arrows - 3:Forward, turn left+forward, turn right+forward - 2:Forward, turn right+forward
    movement_max_actions= 4,
## Complexity of the action space and game abstractions. Default: -1
## -1: Infer. Generally it's the highest implemented settings
## 0: Only directions, bypass powerups actions. Agent must step over checkpoints only
## 1: Adds an interaction button (for checkpoints) and powerups
## 2: Adds NPC
## 3: Adds menu (to implement)
    action_complexity= 2,        
## Format of the screen observation space. Default: -1
## -1: Infer. Generally it's the highest implemented settings
## 0: No screen - 1: Tile matrix - 2: OneHot - 3: Monocromatic-RGB - 4: Assets-RGB
    screen_observation_type= 3,
## Maximum number of steps until the environment is done. Default: 2**31
    max_steps= 2**16,
## Converts the RGB screen observation space to grayscale. Default: False
    grayscale_screen= False,
## Removes the channel axis from grayscale images. Default: True
#    grayscale_single_channel= True,
## Downscale factor of the screen, only with screen_observation_type>=4
#    screen_downscale= 1,          
## Automatically adds the screen to the observations. Default: True
## The screen can always be collected via env.get_screen_observation() following screen_observation_type config
#    auto_screen_obs= True,
## Automatically flattens the observation data to a single-dimension vector. Default: False
#    flat_obs= False,
## Adds a dummy entry in the action space that does nothing. Default: False,    
#    action_nop= False,
### Save-state related
## The event used as starting point once the game is reset. Most of the game state will be infered. Default: ""
## See data/events.json file of the game, or call cli with args "game_name --mode info"
    starting_event= "",
## A list of event names that will be marked as completed on reset. Default []
    starting_collected_flags= [],
## Granularity of the automatic save state routine. Default: 0 (disabled)
## Use env.rewind_state(saved_steps=1) to reload the previous queued state.
#    rewind_steps= 0,
### Screen-debug only
## Saves screen frames for debugging purposes, execution will be slower. Default. False
#    log_screen= False,
## Integer multiplier of the window size exposed to the agent. Default: 1
#    screen_view_mult= 1,
## Custom height of the window size exposed to the agent. Default: 9
#    screen_y= 9,
## Custom width of the window size exposed to the agent. Default: 10
#    screen_x= 10,
)
```

More details on the code and settings, will be provided in future releases.


Deterministic agents
====================
Even if RL agents are supposed to train their policy autonomously, I still added support for deterministic rule-based bots.
Current progress is still not mature enough to have smart agents, but some behaviours can already be expressed, even if there is no certain they will lead to good actions.
In the future, one could implement an hybrid algorithm that uses both policy and determinism (possibly for some pretraining, or tricky spots).
See examples/example_run_deterministic_agent.py to see in action an agents that automatically enters a visible warp.


Next-implementations
====================
Progress will be very erratic and won't follow a strict order. The plan is:

Compatibility
* Support more Python versions

Games
* Complete exploration_world1 main progress (currently at 60%)
* Complete creatures_world1 main progress (halted before 40%)
* Add abstractions for menu handling
* Add abstractions and better data structure for creatures_world1 battle system
* Menu rendering
* Find a way to prevent abuse of Teleport powerup on modes not relying on menu (too disruptive)
* Set an extra frame for warps transitions
* More games archetypes

Core speedup
* Pufferlib example model
* Optimize scripting and NPC handling
* Port relevant game sections to Cython

Environment
* Multi-action wrappers
* Partial state randomization or perturbation
* Improve environment validation for custom games
* Scripting for custom NPC suggesting informative data like coordinates
* Random dummy NPC generation
* Direct text interactions. Data provided as raw string and partially encoded on the screen
* Boilerplate routines for dataset generation, used for offline-learning

Deterministic agents
* Build more fixed algorithm on a per-game basis

Customization
* UI helpers for game editing

Bugfixes
* Fingers crossed!


Special thanks
==============
I want to thank programmers that were really inspiring with their works

* baekalfen - for his awesome PyBoy - [Game Boy emulator written in Python](https://github.com/Baekalfen/PyBoy)
* PWhiddy - for his inspiring PokemonRedExperiments - [Train RL agents to play Pokemon Red](https://github.com/PWhiddy/PokemonRedExperiments/tree/v2-env)
* jsuarez5341 - for his fast PufferLib library - [Simplifying reinforcement learning for complex game environments](https://github.com/PufferAI/PufferLib)
* capnspacehook, CJBoey, leanke, thatguy11325, xinpw8 - for their progress in developing successful models on GB games
