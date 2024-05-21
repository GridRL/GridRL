#!/usr/bin/env python3

"""Main cli for game play and debugging."""

from typing import Union
import sys
import argparse
import numpy as np
sys.dont_write_bytecode=True

#__all__=[]

if __package__ is None or len(__package__)==0:
    from games_list import GAMES_LIST
    from game_module_selector import GameModuleSelector,get_game_default_env_class
    from configs_speedup_dependencies import print_configs_speedup_settings
    from functions_benchmarking import get_benchmark_env_base_configs,benchmark_envs,benchmark_env_multiconfig,profile_env
    from core_environment import make_env,validate_environment
else:
    from gridrl.games_list import GAMES_LIST
    from gridrl.game_module_selector import GameModuleSelector,get_game_default_env_class
    from gridrl.configs_speedup_dependencies import print_configs_speedup_settings
    from gridrl.functions_benchmarking import get_benchmark_env_base_configs,benchmark_envs,benchmark_env_multiconfig,profile_env
    from gridrl.core_environment import make_env,validate_environment

def argparse_cli_game()->argparse.Namespace:
    """Cli parse arguments."""
    parser=argparse.ArgumentParser()
    parser.add_argument("game",type=str,help="Name of the GridRL game.")
    parser.add_argument("--mode",type=str,help="Name of the mode to run.",choices=["map","info","play","sandbox","gif","stream","cythontest","benchmark","menu_benchmark","profile","examples","copy","test"],default="test")
    parser.add_argument("--action_complexity",type=int,help="Complexity of the game and actions performed.",choices=[-1,0,1,2,3,4],default=-1)
    parser.add_argument("--screen_observation_type",type=int,help="Type of observation space returned.",choices=[-1,0,1,2,3,4],default=-1)
    parser.add_argument("--starting_event",type=str,help="Starts from a custom event flag.",default="")
    parser.add_argument("--screen_view_mult",type=int,help="Extends the game screen for debugging purposes.",default=1)
    parser.add_argument("-stdout",action="store_true",help="Avoid redirecting UI stdout to textbox.",default=False)
    return parser.parse_args()

def argparse_cli_nogame()->argparse.Namespace:
    """Cli parse arguments."""
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode",type=str,help="Name of the mode to run.",choices=["test"],default="nop")
    return parser.parse_args()

def args_namespace_to_dict(namespace:Union[argparse.Namespace,dict])->dict:
    """Convertion of argparse.Namespace to dict."""
    return {k:args_namespace_to_dict(v) if isinstance(v,argparse.Namespace) else v for k,v in vars(namespace).items()}

def cli_run_game(game_name:str,config:Union[dict,None]=None,gif:bool=False,streaming:bool=False,validate:bool=True,redirect_stdout:bool=True):
    """Run the game in UI mode given configuration and extra settings."""
    env_config={"action_complexity":3,"screen_observation_type":4}
    if config is not None:
        env_config.update(dict(config))
    env_class=get_game_default_env_class(game_name)
    if env_class is None:
        print(f"Game [{game_name}] class not found.")
        return False
    if gif:
        env_config.update({"log_screen":True})
    if streaming:
        env_config["stream"]=True
    if validate:
        validate_environment(env_class,{"skip_validation":True},fast=True,verbose=True)
    env_config.update({"gui":True,"redirect_stdout":redirect_stdout,"skip_validation":True})
    print(f"Starting game [{game_name}].")
    env=make_env(env_class,rank=0,env_config=env_config,
        agent_class=None,agent_args={},seed=0)()
    env.run_quit()
    return True

def cli_show_map(game_name:str)->None:
    """Show the global map in legacy format."""
    env_config={"action_complexity":3,"screen_observation_type":4,"skip_validation":True}
    env=make_env(game_name,rank=0,env_config=env_config)()
    env.show_global_map()

def cli_show_game_info(game_name:str)->None:
    """Show the global map in legacy format."""
    env_config={"action_complexity":3,"screen_observation_type":4,"skip_validation":True}
    env=make_env(game_name,rank=0,env_config=env_config)()
    validate_environment(env,fast=True,verbose=True)
    print("[Game state]")
    for k,v in env.game_state.items():
        if isinstance(v,np.ndarray):
            print(f"\t{k:31.31}\t{v.dtype}\t{v.shape}")
        else:
            try:
                tshape=len(v)
            except TypeError:
                tshape=1
            print(f"\t{k:31.31}\t{type(v).__name__}\t{tshape}")
    print(f"[Event flags]\n\t{list(env.event_flags_lookup.keys())}")
    env.show_global_map()

def cli_copy_game(game_name:str)->bool:
    """Copy the selected game code to the current directory."""
    gs=GameModuleSelector(game_name)
    print(f"Copying content of game [{game_name}] to [{gs.custom_dir}].")
    ret=gs.copy_game_repo(force=False,verbose=True)
    print(f"\t{'Copied!' if ret else 'Failed.'}")
    return ret

def cli_copy_examples()->bool:
    """Copy examples code to the current directory."""
    gs=GameModuleSelector(GAMES_LIST[0])
    print(f"Copying [examples] to [{gs.selected_base_dir}].")
    ret=gs.copy_examples(force=False,verbose=True)
    print(f"\t{'Copied!' if ret else 'Failed.'}")
    return ret

def cli_run_cython_test(game_name,running_verbose:bool=True)->None:
    """Development cython test routine."""
    print("Running cython test...")
    print_configs_speedup_settings()
    running_verbose=True
    configs_list=[{"screen_observation_type":0,"action_complexity":0}]
    env_class=get_game_default_env_class(game_name)
    steps=int(1e5)
    np.random.seed(7)
    envs=[env_class(get_benchmark_env_base_configs())]
    benchmark_envs(envs,steps=50000,warmup=0,seed=7,show_screen=False,verbose=True)
    print(envs[0].get_debug_text())
    #benchmark_env_multiconfig(env_class,configs_list,steps=steps,warmup=False,seed=7,running_verbose=running_verbose,verbose=True)
    print(np.random.get_state()[1][:4])
    profile_env(env_class(configs_list[0]),steps=50000)

def cli_run_benchmark(game_name,running_verbose:bool=True)->None:
    """Main benchmark routine."""
    print("Running benchmark with various environment configurations...")
    print_configs_speedup_settings()
    running_verbose=True
    configs_list=[]
    configs_list+=[{"screen_observation_type":i,"action_complexity":j} for j in range(4) for i in range(4)]
    configs_list+=[{"screen_observation_type":4,"action_complexity":j,"screen_downscale":2} for j in range(2,4)]
    env_class=get_game_default_env_class(game_name)
    benchmark_env_multiconfig(env_class,configs_list,steps=50000,seed=7,running_verbose=running_verbose,verbose=True)

def cli_run_menu_benchmark(game_name)->None:
    """Menu benchmark routine."""
    env_config=get_benchmark_env_base_configs()
    env_config.update({"screen_observation_type":4,"action_complexity":3,"screen_downscale":2})
    env=make_env(get_game_default_env_class(game_name),rank=0,env_config=env_config)()
    if not env.true_menu or not env.has_menu():
        action_space_size=0
        print("No menu set, likely not programmed or set in configurations.")
    else:
        action_space_size=8
        env.menu.set_force_menu()
        print("Running menu only benchmark, without any render.")
        benchmark_envs([env.menu],steps=50000,action_space_size=8,seed=7,show_screen=False,verbose=True)
        print("Running environment benchmark.")
    benchmark_envs([env],steps=50000,action_space_size=action_space_size,seed=7,show_screen=False,verbose=True)
    print("Profiling the environment.")
    profile_env(env,steps=5000)

def cli_run_profiling(game_name,running_verbose:bool=True)->None:
    """Main profiling routine."""
    print_configs_speedup_settings()
    env_class=get_game_default_env_class(game_name)
    env_config={"screen_observation_type":4,"action_complexity":3,"screen_downscale":2}
    profile_env(env_class(env_config),steps=50000)

def cli_run_test():
    """Test routine for the package."""
    for game_name in GAMES_LIST[:2]:
        print(f"Running test suite on game [{game_name}].")
        cli_run_benchmark(game_name)

def run_mode(game_name:str,mode_name:str,config:Union[dict,None]=None,redirect_stdout:bool=False)->None:
    """Cli mode selection."""
    if config is None:
        config={}
    if mode_name in ["sa","sandbox"]:
        config["sandbox"]=True
    if mode_name in ["map","m"]:
        cli_show_map(game_name)
    elif mode_name in ["info"]:
        cli_show_game_info(game_name)
    elif mode_name in ["p","play","sa","sandbox","s","stream","g","gif"]:
        cli_run_game(game_name,config=config,gif=mode_name[0]=="g",streaming=(mode_name[:2]+"t")[:2]=="st",
            validate=True,redirect_stdout=redirect_stdout)
    elif mode_name in ["c","ct","cythontest"]:
        cli_run_cython_test(game_name)
    elif mode_name in ["b","bench","benchmark"]:
        cli_run_benchmark(game_name)
    elif mode_name in ["mb","menu_benchmark"]:
        cli_run_menu_benchmark(game_name)
    elif mode_name in ["pr","profile"]:
        cli_run_profiling(game_name)
    elif mode_name in ["e","examples"]:
        cli_copy_examples()
    elif mode_name in ["copy"]:
        cli_copy_game(game_name)
    elif mode_name in ["test"]:
        cli_run_test()

def main_cli(custom_args:Union[list,tuple,None]=None)->int:
    """Cli main loop."""
    if isinstance(custom_args,(list,tuple)):
        sys.argv=[str(k) for k in sys.argv[:1]+list(custom_args)]
    ignore_game=False
    try:
        argv=argparse_cli_game()
    except (KeyboardInterrupt,SystemExit):
        try:
            ignore_game=True
            argv=argparse_cli_nogame()
            if argv.mode=="nop":
                return 2
        except (KeyboardInterrupt,SystemExit):
            return 2
    if isinstance(argv.game,(int,str)):
        try:
            idx=int(argv.game)
            if len(GAMES_LIST)>idx>=0:
                argv.game=GAMES_LIST[idx]
        except (ValueError,TypeError):
            pass
    if ignore_game or argv.game in GAMES_LIST:
        run_mode(game_name=GAMES_LIST[0] if ignore_game else argv.game,config=args_namespace_to_dict(argv),
            mode_name=argv.mode,redirect_stdout=not argv.stdout)
    else:
        print(f"Please select a valid game: [{', '.join(GAMES_LIST)}].")
        return 2
    return 1
