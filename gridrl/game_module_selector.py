#!/usr/bin/env python3

"""Struct for data and code import."""

from typing import Union,Callable
import importlib
import sys
import os
import re
import numpy as np
sys.dont_write_bytecode=True

if __package__ is None or len(__package__)==0:
    from games_list import GAMES_LIST
    from functions_data import (read_map_data_file,read_teleport_data_file,
        read_events_file,read_scripts_file,read_npcs_file,read_powerups_file,
        read_map_png,read_sprites_png,copy_dir
    )
else:
    from gridrl.games_list import GAMES_LIST
    from gridrl.functions_data import (read_map_data_file,read_teleport_data_file,
        read_events_file,read_scripts_file,read_npcs_file,read_powerups_file,
        read_map_png,read_sprites_png,copy_dir
    )

__all__=["GAMES_LIST","GameModuleSelector","get_game_default_env_class"]

LOADED_FROM_MODULE=str(__package__).split(".",maxsplit=1)[0]=="gridrl"
repo_script_dir=f"{os.path.dirname(os.path.realpath(__file__))}{os.sep}"

class GameModuleSelector:
    """Class for selecting game data and implementing local game editing in the [custom_game] folder."""
    def __init__(self,game_name:str,cur_dir:Union[str,None]=None,ignore_custom:bool=False,abort_bad_name:bool=False)->None:
        """Constructor."""
        self.from_module=LOADED_FROM_MODULE
        self.game_modules={k:[] for k in ["game","scripting","env"]}
        self.game_dirs=[]
        self.game_name=str(game_name).lower()
        self.game_class_name=re.sub("[^A-Za-z0-9]","",self.game_name.title())
        if not self.is_valid_game_name(self.game_name):
            if abort_bad_name:
                raise NameError(f"No game found with name [{self.game_name}].")
            self.game_name=str(GAMES_LIST[0])
        game_repo_part_dir=f"games{os.sep}{self.game_name}{os.sep}"
        self.selected_base_dir=f"{os.path.realpath(sys.path[0] if cur_dir is None or not os.path.exists(cur_dir) else cur_dir)}{os.sep}"
        self.game_repo_dir=f"{repo_script_dir}{game_repo_part_dir}"
        self.custom_game_dir=f"{self.selected_base_dir}custom_game{os.sep}{game_repo_part_dir}"
        if not ignore_custom and os.path.exists(self.custom_game_dir):
            for k,_ in self.game_modules.items():
                self.game_modules[k].append(f"custom_game.{k}")
            self.game_dirs.append(self.custom_game_dir)
        root_module=f"{str(__package__).split('.',maxsplit=1)[0]}." if LOADED_FROM_MODULE else ""
        for k,_ in self.game_modules.items():
            self.game_modules[k].append(f"{root_module}games.{self.game_name}.{k}")
        self.game_dirs.append(self.game_repo_dir)
        self.is_custom_game=len(self.game_dirs)>1
        temp_class=self.fallback_get_game_class()
        self.has_encoded_assets=temp_class is not None and bool(temp_class.has_encoded_assets)
    def is_valid_game_name(self,name:str)->bool:
        """Check if the game matches the list of valid games."""
        return name in GAMES_LIST
    def fallback_get_module_by_name(self,name:str)->Callable:
        """Returns the python scope of the game module matching the name."""
        for module_import_name in self.game_modules.get(name,[]):
            try:
                return importlib.import_module(module_import_name)
            except ModuleNotFoundError:
                pass
        return None
    def fallback_get_module_class_by_name(self,name:str)->Callable:
        """Returns the python scope of the class inside the game module matching the name."""
        module=self.fallback_get_module_by_name(name)
        if module is not None:
            try:
                return getattr(module,f"{self.game_class_name}{name.title()}")
            except AttributeError:
                pass
        return None
    def fallback_get_constants_module(self)->Callable:
        """Returns the python scope of the game constants module."""
        return self.fallback_get_module_by_name("constants")
    def fallback_get_game_module(self)->Callable:
        """Returns the python scope of the game logic module."""
        return self.fallback_get_module_by_name("game")
    def fallback_get_scripting_module(self)->Callable:
        """Returns the python scope of the game scripting module."""
        return self.fallback_get_module_by_name("scripting")
    def fallback_get_env_module(self)->Callable:
        """Returns the python scope of the game env module."""
        return self.fallback_get_module_by_name("env")
    def fallback_get_game_class(self)->Callable:
        """Returns the python class of the game environment."""
        return self.fallback_get_module_class_by_name("game")
    def fallback_get_env_class(self)->Callable:
        """Returns the python class of the environment."""
        return self.fallback_get_module_class_by_name("env")
    def fallback_get_game_extra_data_functions(self)->list:
        """Returns a list of extra functions for data handling."""
        game_class=self.fallback_get_game_class()
        return [] if game_class is None else game_class.define_game_extra_data_functions(game_class)
    def fallback_read_single_data_file_from_func(self,func:Callable,only_first:bool=False)->dict:
        """Read single game data file from a callable."""
        for main_parent in self.game_dirs[slice(0,1 if only_first else None)]:
            fdict=func(dir_name="data",main_parent=main_parent)
            if len(fdict)>0:
                return fdict
        return {}
    def fallback_write_single_data_file_from_func(self,func:Callable,obj:dict,from_repo:bool=False)->bool:
        """Write single game data file from a callable."""
        return func(dir_name="data",main_parent=self.game_dirs[-1 if from_repo else 0],obj=obj)
    def fallback_read_all_data_files(self,only_first:bool=False)->dict:
        """Read all game data files."""
        data={}
        for func in [read_map_data_file,read_teleport_data_file,
            read_events_file,read_scripts_file,read_npcs_file,
            read_powerups_file]+self.fallback_get_game_extra_data_functions():
            for i,main_parent in enumerate(self.game_dirs[slice(0,1 if only_first else None)]):
                fdict=func(dir_name="data",main_parent=main_parent)
                if len(fdict)==0:
                    if i>0:
                        print("Couldn't load all data files inside [data] folder.")
                        assert len(fdict)>0
                else:
                    data.update(fdict)
                    break
        assert len(data)>0
        return data
    def fallback_read_image_func(self,image_func:Callable,only_first:bool=False)->np.ndarray:
        """Call a function that reads an image file."""
        for main_parent in self.game_dirs[slice(0,1 if only_first else None)]:
            if not callable(image_func):
                continue
            img=image_func(dir_name="assets",main_parent=main_parent,encoded=self.has_encoded_assets)
            if np.prod(img.shape[:2])>4096:
                return img
        return np.zeros((16,16,3),dtype=np.uint8,order="C")
    def fallback_read_map_png(self)->np.ndarray:
        """Read the global game image."""
        return self.fallback_read_image_func(read_map_png)
    def fallback_read_sprites_png(self)->np.ndarray:
        """Read the sprites image."""
        return self.fallback_read_image_func(read_sprites_png)
    def copy_game_repo(self,force:bool=False,verbose:bool=True)->bool:
        """Copy the selected game code to the current directory."""
        ret=True
        overwrite=force
        for game_dir in ["data","envs","game"]:
            src_dir_name=f"{self.game_repo_dir}{game_dir}{os.sep}"
            dest_dir_name=f"{self.custom_game_dir}{game_dir}{os.sep}"
            if src_dir_name==dest_dir_name:
                return True
            if not overwrite and os.path.isdir(dest_dir_name):
                if verbose:
                    if input("\tThe destination folder already exists. Overwrite it? >> [Y/N] ").lower()!="y":
                        return False
                    overwrite=True
                else:
                    return False
            if not copy_dir(src_dir_name,dest_dir_name,force=True,verbose=False):
                ret=False
        return ret
    def copy_examples(self,force:bool=False,verbose:bool=True)->bool:
        """Copy examples code to the current directory."""
        src_dir_name=f"{self.game_repo_dir}examples{os.sep}"
        dest_dir_name=f"{self.selected_base_dir}examples{os.sep}"
        return copy_dir(src_dir_name,dest_dir_name,force=force,verbose=verbose)

def get_game_default_env_class(name:str):
    """Returns the python class of the environment matching game name."""
    return GameModuleSelector(name).fallback_get_env_class()

if __name__=="__main__":
    assert get_game_default_env_class(GAMES_LIST[0]) is not None
