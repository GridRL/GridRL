#!/usr/bin/env python3

"""Benchmarking utility functions."""

from typing import Any,Callable
import sys
import time
import numpy as np
sys.dont_write_bytecode=True

try:
    from pyinstrument import Profiler
    PYINSTRUMENT_PROFILER=True
except (ModuleNotFoundError,ImportError):
    PYINSTRUMENT_PROFILER=False

if __package__ is None or len(__package__)==0:
    from functions_data import script_dir
else:
    from gridrl.functions_data import script_dir

def get_benchmark_env_base_configs()->dict:
    """Get default settings for the benchmarked environment."""
    return {"movement_max_actions":4,"max_steps":-1,"action_nop":False,
        "validate_environment":False,"log_screen":False,"hsv_image":False}

def bench_core_with_actions_internal(env:Any,actions:np.ndarray,verbose:bool=True)->bool:
    """Core benchmarking running run_action_on_emulator from predefined actions."""
    if hasattr(env,"run_action_on_emulator"):
        try:
            for a in actions:
                env.run_action_on_emulator(a)
                env.step_count+=1
        except NotImplementedError:
            return False
    elif hasattr(env,"force_menu") and hasattr(env,"menu_state_menu") and hasattr(env,"step"):
        try:
            for a in actions:
                env.step(a)
        except NotImplementedError:
            return False
    else:
        if verbose:
            print("Environment misses [run_action_on_emulator] method.")
        time.sleep(0.1)
    return True

def bench_step_with_actions_internal(env:Any,actions:np.ndarray,verbose:bool=True)->bool:
    """Environment benchmarking running step from predefined actions."""
    for a in actions:
        try:
            step_data=env.step(a)
            if step_data is not None and step_data[2]:
                env.reset()
        except NotImplementedError:
            if verbose:
                print("Environment misses [step] or [reset] methods.")
            return False
    return True

def bench_step_predict_internal(env:Any,total_steps:int,verbose:bool=True,**kwargs)->bool:
    """Environment benchmarking running step predicting actions."""
    for _ in range(total_steps):
        try:
            step_data=env.step(env.predict_action())
            if step_data is not None and step_data[2]:
                env.reset()
        except NotImplementedError:
            if verbose:
                print("Environment misses [step] or [reset] methods.")
            return False
    return True

def benchmark_envs(envs:list,steps:int=50000,warmup:int=10000,action_space_size:int=0,
    seed:int=7,show_screen:bool=True,verbose:bool=True)->list:
    """Benchmark a list of environments."""
    bench_envs=envs if isinstance(envs,(list,tuple)) else [envs]
    if verbose and bench_envs[0].log_screen:
        print("Screen logging is active, the benchmark will be significantly slower.")
    for env in bench_envs:
        env.reset(seed=seed)
        if warmup>100:
            bench_core_with_actions_internal(env,np.zeros((warmup,),dtype=np.uint8,order="C"))
    total_steps=max(1000,steps)
    bench_funcs=[bench_core_with_actions_internal]+[bench_step_with_actions_internal
        for i in range(len(bench_envs))]
    bench_names=["CORE-EMU-ENV1"]+[f"ENV{i+1:d}-STEP" for i in range(len(bench_envs))]
    bench_env_idxs=[0]+list(range(len(bench_envs)))
    bench_results=[]
    used_action_space_size=max(1,bench_envs[0].movement_max_actions if action_space_size<2 else action_space_size)
    actions_emu=np.random.randint(0,used_action_space_size,total_steps)
    actions_steps=[np.random.randint(0,used_action_space_size,total_steps)
        for env in bench_envs]
    for i,(bfunc,bname,env_idx) in enumerate(zip(bench_funcs,bench_names,bench_env_idxs)):
        bench_envs[env_idx].reset(seed=seed)
        bench_time=time.time()
        bench_envs[env_idx].benchmarking=True
        bench_success=bfunc(bench_envs[env_idx],actions_emu if i==0 else actions_steps[env_idx],verbose)
        bench_envs[env_idx].benchmarking=False
        bench_time=max(1e-6,time.time()-bench_time) if bench_success else 1e6
        bench_results.append(bench_time)
        if verbose:
            ratio=bench_results[0]/bench_time
            sps=total_steps/bench_time
            bstr=f"\tvs CORE: {ratio:-6.1%} (x{1./ratio:4.2f})" if i>0 else ""
            print(f"{bname}\tSPS: {sps:-7.0f}\tTime: {bench_time:-5.1f} s{bstr}")
    if show_screen:
        bench_envs[0].show_agent_map()
        bench_envs[0].show_agent_screen()
    if bench_envs[0].log_screen:
        if verbose and steps>2000:
            print("\n\tGenerating GIF, it may take a while...")
        bench_envs[0].save_gif(f"{script_dir}benchmark_run_{int(time.time()):d}.gif",speedup=16)
    elif verbose:
        print("\n\tTo save a GIF, set [log_screen: True] in the environment config dictionary.")
    return bench_results

def benchmark_agents(env,agents:list,steps=50000,warmup=10000,
    seed=7,verbose:bool=True)->list:
    """Benchmark a list of agents on the environment."""
    env.reset(seed=seed)
    if warmup>100:
        for act in env.roll_random_actions_without_nop(warmup):
            env.step(act)
    total_steps=max(1000,steps)
    bench_agents=list(agents)
    bench_names=[k.__class__.__name__ for k in bench_agents]
    bench_results=[]
    for bagent,bname in zip(bench_agents,bench_names):
        env.reset(seed=seed)
        bagent.set_env(env)
        bench_time=time.time()
        bench_success=bench_step_predict_internal(env,steps,verbose)
        bench_time=max(1e-6,time.time()-bench_time) if bench_success else 1e6
        bench_results.append(bench_time)
        if verbose:
            ratio=bench_results[0]/bench_time
            sps=total_steps/bench_time
            print(f"{bname}\tSPS: {sps:.0f}\tTime: {bench_time:.2f} s"
                f"\t| {env.get_coordinates_text()} {env.get_collected_flags_names()}")
    for i in range(1,len(bench_results)):
        ratio=bench_results[0]/bench_results[i]
        print(f"\tRATIO {bench_names[0]}/{bench_names[i]}: {ratio:.2%} ({1./ratio:.2f})")
    return bench_results

def benchmark_env_multiconfig(env_class,configs_list:list,steps:int=50000,
    seed:int=7,running_verbose:bool=True,verbose:bool=True)->list:
    """Compare benchmark time of different environment configurations."""
    if verbose and not running_verbose and len(configs_list)>1:
        print("Waiting for all configurations tests...")
    total_steps=max(1000,steps)
    base_configs=get_benchmark_env_base_configs()
    envs=[env_class({**base_configs,**conf}) for conf in configs_list]
    bench_results=benchmark_envs(envs,steps=total_steps,warmup=500,
        seed=seed,show_screen=False,verbose=running_verbose)
    if verbose:
        bench_str_results=[[bt,"{:-7.0f}\t{}\t{}".format(total_steps/bt,
            "\t" if i==0 else f"{bench_results[0]/bt:-6.1%}\tx{bt/bench_results[0]:4.2f}",conf)]
                for i,(bt,conf) in enumerate(zip(bench_results,[{"core_emu":1}]+configs_list))]
        print("\t".join([k.rjust(6) for k in ["SPS","%Core","(xInv)","Configs"]]))
        print("\n".join([k[1] for k in sorted(bench_str_results,key=lambda x:x[0])]))
    return bench_results

def profiler_function_monitor(func:Callable):
    """Count function calls."""
    def decorating_function(*args,**kwargs):
        """Decorator."""
        decorating_function.calls+=1
        return func(*args,**kwargs)
    decorating_function.calls=0
    return decorating_function

def get_env_methods_names(env:Any)->list:
    """Return a list with all environment method names."""
    return [k for k in dir(env) if not k.startswith("__") and not k.endswith("__") and callable(getattr(env,k))]

def wrap_env_with_profiler_function_monitor(env:Any)->Any:
    """Wrap all environment methods to count calls."""
    method_names=get_env_methods_names(env)
    for m in method_names:
        setattr(env,m,profiler_function_monitor(getattr(env,m)))
    return env

def print_wrapped_profiler_results(env:Any,steps:int)->Any:
    """Print calls results of a wrapped environment."""
    method_names=get_env_methods_names(env)
    sorted_calls_data=sorted([[m,getattr(env,m).calls] for m in method_names if hasattr(getattr(env,m),"calls")],key=lambda x:(-x[1],x[0]))
    for scd in sorted_calls_data:
        if scd[1]<1:
            break
        print(f"\t{scd[1]:d}\t{float(scd[1])/max(1,steps):.1%}\t{scd[0]}")

def profile_env(env,steps:int=50000,interval:float=0.0001,fallback_banchmark:bool=True)->None:
    """Run pyinstrument profile."""
    if PYINSTRUMENT_PROFILER:
        actions=np.random.randint(0,env.movement_max_actions+1,max(100,steps))
        copy_env=env.deepcopy()
        with Profiler(interval=interval) as profiler:
            for act in actions:
                copy_env.step(act)
        profiler.print()
        print("Repeating to count functions calls.")
        copy_env=wrap_env_with_profiler_function_monitor(copy_env)
        actions=actions[:actions.shape[0]//20]
        for act in actions:
            copy_env.step(act)
        print_wrapped_profiler_results(copy_env,actions.shape[0])
    else:
        print("Couldn't load [pyinstrument] package to profile. Run pip install pyinstrument.")
        if fallback_banchmark:
            print("Running benchmark instead.")
            benchmark_envs(env,steps,show_screen=False,verbose=True)
