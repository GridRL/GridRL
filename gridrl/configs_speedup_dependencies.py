#!/usr/bin/env python3

"""Definition of constants and wrappers for caching and numbda speedup ."""

from typing import Callable
from functools import lru_cache#,partial
import sys
#import os
sys.dont_write_bytecode=True

SPEEDUP_CACHE=True
SPEEDUP_CYTHON=False
SPEEDUP_JAX=False
SPEEDUP_NUMBA=False

#if SPEEDUP_JAX:
#    os.environ["JAX_TRACEBACK_FILTERING"]="off"
#    try:
#        from jax import jit as j_jit,numpy as j_np
#    except (ModuleNotFoundError,ImportError):
#        SPEEDUP_JAX=False
#        print("Couldn't load [jax] package. The environment could be slower.")
#if SPEEDUP_JAX:
#    SPEEDUP_NUMBA=False
#elif SPEEDUP_NUMBA:
#    try:
#        from numba import jit as nb_jit,uint16 as nb_uint16
#    except (ModuleNotFoundError,ImportError):
#        SPEEDUP_NUMBA=False
#        print("Couldn't load [numba] package. The environment could be slower.")
#if not SPEEDUP_JAX:
#    j_np=np
if not SPEEDUP_NUMBA:
    (nb_uint16,nb_prange)=(int,range)

def no_lru_cache_func(maxsize:int=128)->Callable:
    """Placeholder wrapper for testing disabled cache."""
    def decorating_function(user_function:Callable)->Callable:
        """Decorator."""
        def wrapper(*args,**kwargs)->Callable:
            """Wrapper."""
            return user_function(*args,**kwargs)
        return wrapper
    return decorating_function

def no_j_jit_func(**kwargs)->Callable:
    """Placeholder wrapper for missing jax package."""
    def decorating_function(user_function:Callable)->Callable:
        """Decorator."""
        def wrapper(*args,**kwargs)->Callable:
            """Wrapper."""
            return user_function(*args,**kwargs)
        return wrapper
    return decorating_function

def no_nb_jit_func(func_types:str="uint8(uint8)",**kwargs)->Callable:
    """Placeholder wrapper for missing numba package."""
    def decorating_function(user_function:Callable)->Callable:
        """Decorator."""
        def wrapper(*args,**kwargs)->Callable:
            """Wrapper."""
            return user_function(*args,**kwargs)
        return wrapper
    return decorating_function

lru_cache_func=no_lru_cache_func if not SPEEDUP_CACHE else lru_cache
nb_jit_func=no_nb_jit_func# if not SPEEDUP_NUMBA else nb_jit
j_jit_func=no_j_jit_func# if not SPEEDUP_JAX else j_jit
j_jit_mfunc=no_j_jit_func# if not SPEEDUP_JAX else partial(j_jit,static_argnums=(0,))

def print_configs_speedup_settings():
    """Prints the type of optimizations applied, also using third-party packages."""
    print(f"Cache: {SPEEDUP_CACHE}\tCython: {SPEEDUP_CYTHON}"
        f"\tJAX: {SPEEDUP_JAX}\tNumba: {SPEEDUP_NUMBA}")

def dummy_pep_pass()->None:
    """Dummy function to pass pep validation."""
    @no_lru_cache_func(maxsize=1)
    def dummy_cached_func()->None:
        """Dummy cached function."""
        return 0
    @no_nb_jit_func(func_types="boolean()",nopython=True)
    def dummy_numba_func()->bool:
        """Dummy numba function."""
        return True
    @no_j_jit_func()
    def dummy_jax_func()->bool:
        """Dummy jax function."""
        return True
    dummy_cached_func()
    dummy_numba_func()
    dummy_jax_func()

if __name__=="__main__":
    print_configs_speedup_settings()
    dummy_pep_pass()
