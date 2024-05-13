#!/usr/bin/env python3

"""Data utility functions."""

from typing import Union,Any,Callable
from functools import partial
from collections import OrderedDict,defaultdict
import sys
import os
import shutil
import json
import zlib
import gzip
import pickle
import numpy as np
from PIL import Image
sys.dont_write_bytecode=True

if __package__ is None or len(__package__)==0:
    from core_constants import (MAP_DATA_FILENAME,MAP_METADATA_FILENAME,
        TELEPORT_DATA_FILENAME,EVENTS_FILENAME,SCRIPTS_FILENAME,NPCS_FILENAME,POWERUPS_FILENAME,
        MAP_PNG_FILENAME,SPRITES_PNG_FILENAME,
    )
else:
    from gridrl.core_constants import (MAP_DATA_FILENAME,MAP_METADATA_FILENAME,
        TELEPORT_DATA_FILENAME,EVENTS_FILENAME,SCRIPTS_FILENAME,NPCS_FILENAME,POWERUPS_FILENAME,
        MAP_PNG_FILENAME,SPRITES_PNG_FILENAME,
    )

#__all__=[]


try:
    script_dir=f"{os.path.dirname(os.path.realpath(__file__))}{os.sep}"
except NameError:
    script_dir=f".{os.sep}"
data_dir=f"{os.path.dirname(os.path.realpath(__file__))}{os.sep}games{os.sep}exploration_world1{os.sep}"

def get_function_from_self_or_scope(func_name:str,self:Any=None,scope_fn:Any=None)->Callable:
    """Function to extract a callable by name, from an object, module or global."""
    func=None
    if not isinstance(func_name,str):
        return False
    if hasattr(self,func_name):
        func=getattr(self,func_name)
    if not callable(func) and scope_fn is not None:
        if hasattr(scope_fn,func_name):
            func=getattr(scope_fn,func_name)
        if not callable(func):
            if func_name in globals().keys():
                func=globals()[func_name]
                if callable(func):
                    func=partial(func,self)
        else:
            func=partial(func,self)
    return func

def pickle_load(file_path:str)->dict:
    """Load a state from a pickle dumped file."""
    data=None
    if os.path.isfile(file_path):
        with open(file_path,mode="rb") as f:
            try:
                data=pickle.load(f)
            except (pickle.UnpicklingError,FileNotFoundError,OSError):
                pass
    return data

def pickle_save(file_path:str,obj)->bool:
    """Save a state into a pickle dump file."""
    with open(file_path,mode="wb") as f:
        try:
            pickle.dump(obj,f)
            return True
        except (TypeError,FileNotFoundError,OSError):
            pass
    return False

def decode_image(enc_img:np.ndarray,reserved_byte:int=0xF9,xor_byte1:int=0xC5,xor_byte2:int=0xA1)->np.ndarray:
    """Decode an encoded image."""
    try:
        dec=np.frombuffer(zlib.decompress(np.ascontiguousarray(enc_img,dtype=np.uint8)),dtype=np.uint8)
    except (TypeError,zlib.error):
        return np.zeros((16,16,3),dtype=np.uint8,order="C")
    if len(dec)>13 and dec[-13]==reserved_byte:
        try:
            img=np.ascontiguousarray(np.bitwise_xor(xor_byte1,dec[:-13]).reshape(
                np.frombuffer(np.bitwise_xor(xor_byte2,dec[-12:]),dtype=np.uint32).tolist()
            ))
            if len(img.shape)<2:
                raise ValueError
            if len(img.shape)==2:
                img=np.expand_dims(img,axis=-1)
            if img.shape[2]==1:
                img=np.repeat(img,3,axis=2)
            return img
        except ValueError:
            pass
    return np.zeros((16,16,3),dtype=np.uint8,order="C")

def format_child_filename_path(file_name:str,dir_name:str="data",main_parent:str=data_dir)->str:
    """Format a file path given a parent directory."""
    return f"{main_parent}{dir_name}{os.sep}{file_name}"

def map_dict_data_casting(data:dict,file_data:dict,dict_casting:dict)->dict:
    """Input data type casting."""
    for k,v in file_data.items():
        if k=="maps":
            assert isinstance(v,dict)
            data[k]={convert_hex_str(mid):np.asarray(v[mid],dtype=np.uint8) for mid in v.keys()}
        elif k=="maps_names":
            data[k]={convert_hex_str(mid):v[mid] for mid in v.keys()}
        elif isinstance(v,dict):
            real_keys=list(v.keys())
            idx_keys=[convert_hex_str(idx) for idx in real_keys]
            arr_size=[0 if len(idx_keys)==0 else np.max(idx_keys)+1]
            assert len(real_keys)>0 and (np.min(idx_keys)>=0 and arr_size[0]<65500)
            if isinstance(v[real_keys[0]],(list,set,tuple,np.ndarray)):
                arr_size.append(len(v[real_keys[0]]))
                if (k=="warps" or "2d" in k) and arr_size[-1]>0 and isinstance(v[real_keys[0]][0],(list,set,tuple,np.ndarray)):
                    arr_size.append(len(v[real_keys[0]][0]))
            data[k]=np.zeros(tuple(arr_size),dtype=dict_casting.get(k,np.int16),order="C")
            for rk,idx in zip(real_keys,idx_keys):
                data[k][idx]=v[rk]
        else:
            data[k]=np.array(v,dtype=dict_casting.get(k,np.int16))
    return data

def is_json_key_comment(name)->bool:
    """Check for special comment rows."""
    lower_name=str(name).lower()
    for k in ["comment","description","skip","#"]:
        if lower_name.startswith(k):
            return True
    return False

def filter_dict_key(data:dict,func:Union[Callable,None]=None)->dict:
    """Strip dict keys given a callable."""
    return {str(k).rsplit("|",maxsplit=1)[-1]:v for k,v in data.items() if func(k)} if callable(func) else {str(k).rsplit("|",maxsplit=1)[-1]:v for k,v in data.items()}

def filter_dict_key_depending_on_npc(data:dict,npc:bool=True)->dict:
    """Strip dict key if they are not supposed to be used in NPC mode."""
    return filter_dict_key(data,(lambda x:"nonpc" not in str(x).lower()) if npc else None)

def convert_hex_str(val)->int:
    """String int casting, supporting hex."""
    return int(val) if not isinstance(val,str) or "x" not in val.lower() else int(val,16)

class NumpyJsonEncoder(json.JSONEncoder):
    """JSON type encoder."""
    def default(self,o:object):
        """Object convertion to primitive types."""
        if o is None or isinstance(o,(list,dict)):
            return o
        if isinstance(o,(set,tuple)):
            return list(o)
        if isinstance(o,(OrderedDict,defaultdict)):
            return dict(o)
        if isinstance(o,np.ndarray):
            return o.tolist()
        if isinstance(o,(float,np.float_,np.float16,np.float32,np.float64)):
            return float(o)
        if isinstance(o,(int,np.int_,np.intc,np.intp,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64)):
            return int(o)
        if isinstance(o,complex):
            return float(o.imag)
        return json.JSONEncoder.default(self,o)

def read_json_file(file_path:str,check_uncompressed:bool=True)->dict:
    """Read json content from a file."""
    file_data={}
    checked_paths=[file_path]
    if check_uncompressed and file_path.endswith(".gz"):
        checked_paths.append(file_path[:-3])
    for tpath in checked_paths:
        if os.path.isfile(tpath):
            read_func=gzip.open if tpath.endswith(".gz") else open
            with read_func(tpath,mode="rt") as f:
                try:
                    file_data=json.load(f)
                except (FileNotFoundError,json.decoder.JSONDecodeError,
                       UnicodeDecodeError,TypeError,OSError):
                    print(f"Error reading json file [{tpath}].")
                    raise
            if len(file_data)>0:
                break
    return file_data

def write_json_file(file_path:str,obj:Union[dict,None],formatted:bool=False,cls:Callable=NumpyJsonEncoder)->dict:
    """Save objects to a file in json format."""
    if obj is None:
        return False
    ret=False
    if os.path.exists(file_path):
        os.unlink(file_path)
    read_func=gzip.open if file_path.endswith(".gz") else open
    with read_func(file_path,mode="wt") as f:
        try:
            extra_args={"indent":4,"sort_keys":True,"separators":(',', ': ')} if formatted else {}
            json.dump(obj,f,cls=cls,**extra_args)
            ret=True
        except (FileNotFoundError,UnicodeDecodeError,TypeError,OSError):
            print(f"Error writing to json file [{file_path}]")
            raise
    return ret

def read_image_rgb(file_path:str,encoded:bool=False)->np.ndarray:
    """Read an image file."""
    if os.path.isfile(file_path):
        try:
#            if encoded:
#                return decode_image(np.load(file_path,allow_pickle=False))
            img=np.array(Image.open(file_path).convert("RGB"),dtype=np.uint8)
            if encoded:
                img=decode_image(img)
            return img
        except (FileNotFoundError,OSError):
            pass
    return np.zeros((16,16,3),dtype=np.uint8,order="C")

def read_map_data_file(dir_name:str="data",main_parent:str=data_dir)->dict:
    """Read the map data file."""
    data={}
    file_data=read_json_file(format_child_filename_path(MAP_DATA_FILENAME,dir_name,main_parent))
    if isinstance(file_data,dict) and len(file_data)>0:
        dict_casting={"channels":np.uint8,"sizes":np.int16,
            "legacy_global_starting_coords":np.int16,"warps":np.int16,
            "warps_count":np.uint8,"bounds":np.int16,"connections_mask":np.uint8}
        data=map_dict_data_casting(data,file_data,dict_casting)
    return data

def read_map_metadata_file(dir_name:str="data",main_parent:str=data_dir)->dict:
    """Read the map meta-data file."""
    data={}
    file_data=read_json_file(format_child_filename_path(MAP_METADATA_FILENAME,dir_name,main_parent))
    if isinstance(file_data,dict) and len(file_data)>0:
        dict_casting={"global_starting_coords":np.int16}
        file_data={int(e["id"]):e["coordinates"][:2][::-1] for e in file_data["regions"] if int(e["id"])>=0}
        for i in range(0x100):
            if i not in file_data.keys():
                file_data[i]=[0,0]
        file_data={"global_starting_coords":[k[1] for k in sorted([[k,v] for k,v in file_data.items()],key=lambda x:x[0])]}
        data=map_dict_data_casting(data,file_data,dict_casting)
    return data

def read_teleport_data_file(dir_name:str="data",main_parent:str=data_dir)->dict:
    """Read the teleport locations file."""
    data={}
    file_data=read_json_file(format_child_filename_path(TELEPORT_DATA_FILENAME,dir_name,main_parent))
    if isinstance(file_data,dict) and len(file_data)>0:
        data["teleport_data"]={convert_hex_str(k):[np.array([convert_hex_str(s) for s in v[0][:2]],dtype=np.int16),v[1]] for k,v in file_data.items()
            if not is_json_key_comment(k) and isinstance(v,(list,set,tuple,np.ndarray)) and len(v)>1 and
            isinstance(v[0],(list,set,tuple,np.ndarray)) and len(v[0])>1 and isinstance(v[1],str)
        }
    return data

def read_events_file(dir_name:str="data",main_parent:str=data_dir)->dict:
    """Read the events file."""
    data={}
    file_data=read_json_file(format_child_filename_path(EVENTS_FILENAME,dir_name,main_parent))
    if isinstance(file_data,dict) and len(file_data)>0:
        data["events"]=OrderedDict([[k,[np.array([convert_hex_str(s) for s in v[0][:4]],dtype=np.int16),np.array([convert_hex_str(s) for s in v[1][:3]],dtype=np.int16),np.array([convert_hex_str(s) for s in v[2][:6]],dtype=np.int16).clip(0,100),[str(s) for s in v[3][:9]]]] for k,v in file_data.items()
            if not is_json_key_comment(k) and isinstance(v,(list,set,tuple,np.ndarray)) and len(v)>3 and
            isinstance(v[0],(list,set,tuple,np.ndarray)) and len(v[0])>3 and isinstance(v[1],(list,set,tuple,np.ndarray)) and len(v[1])>0 and len(v[2])>0 and isinstance(v[3],(list,set,tuple,np.ndarray))
        ])
    return data

def read_scripts_file(dir_name:str="data",main_parent:str=data_dir)->dict:
    """Read the scripts file."""
    data={}
    file_data=read_json_file(format_child_filename_path(SCRIPTS_FILENAME,dir_name,main_parent))
    if isinstance(file_data,dict) and len(file_data)>0:
        data["scripts"]={k:[np.array([convert_hex_str(s) for s in v[0][:3]],dtype=np.int16),v[1]] for k,v in file_data.items()
            if not is_json_key_comment(k) and isinstance(v,(list,set,tuple,np.ndarray)) and len(v)>1 and
            isinstance(v[0],(list,set,tuple,np.ndarray)) and len(v[0])>2 and isinstance(v[1],str)
        }
    return data

def read_npcs_file(dir_name:str="data",main_parent:str=data_dir)->dict:
    """Read the npcs file."""
    data={}
    file_data=read_json_file(format_child_filename_path(NPCS_FILENAME,dir_name,main_parent))
    if isinstance(file_data,dict) and len(file_data)>0:
        data["npcs"]={k:[np.array([convert_hex_str(s) for s in v[0][:8]],dtype=np.int16),v[1]] for k,v in file_data.items()
            if not is_json_key_comment(k) and isinstance(v,(list,set,tuple,np.ndarray)) and len(v)>1 and
            isinstance(v[0],(list,set,tuple,np.ndarray)) and len(v[0])>7 and isinstance(v[1],(str,list,set,tuple))
        }
    return data

def read_powerups_file(dir_name:str="data",main_parent:str=data_dir)->dict:
    """Read the powerups file."""
    data={}
    file_data=read_json_file(format_child_filename_path(POWERUPS_FILENAME,dir_name,main_parent))
    if isinstance(file_data,dict) and len(file_data)>0:
        data["powerups"]={k:[str(s) for s in v[:9]] for k,v in file_data.items() if not is_json_key_comment(k) and isinstance(v,(list,set,tuple,np.ndarray))}
    return data

def read_all_data_files(dir_name:str="data",main_parent:str=data_dir,skip_assert:bool=False)->dict:
    """Read all game data files."""
    data={}
    for func in [read_map_data_file,read_map_metadata_file,read_teleport_data_file,read_events_file,read_scripts_file,read_npcs_file,read_powerups_file]:
        fdict=func(dir_name,main_parent=main_parent)
        if len(fdict)==0:
            print(f"Couldn't load all data files inside [{dir_name}] folder.")
        if not skip_assert:
            assert len(fdict)>0
        data.update(fdict)
    return data

def read_map_png(dir_name:str="assets",main_parent:str=data_dir,encoded:bool=False)->np.ndarray:
    """Read the global map image."""
    return read_image_rgb(format_child_filename_path(MAP_PNG_FILENAME,dir_name,main_parent),encoded=encoded)

def read_sprites_png(dir_name:str="assets",keep_alpha:bool=False,main_parent:str=data_dir,encoded:bool=False)->np.ndarray:
    """Read the sprites image."""
    alpha_color=[0xFF,0x7F,0x7F]
    img=read_image_rgb(format_child_filename_path(SPRITES_PNG_FILENAME,dir_name,main_parent),encoded=encoded)
    img[np.all(img[:,:,:len(alpha_color)]==alpha_color,axis=-1)]=0xFF
    return img if keep_alpha else np.ascontiguousarray(img[:,:,:3])

def write_map_data_file(dir_name:str="data",main_parent:str=data_dir,obj:Union[dict,None]=None)->bool:
    """Write the map data file."""
    return write_json_file(format_child_filename_path(MAP_DATA_FILENAME,dir_name,main_parent),obj,formatted=True)

def copy_dir(src_dir_path:str,dest_dir_path:str,force:bool=False,verbose:bool=True)->bool:
    """Copy folders data."""
    if src_dir_path==dest_dir_path:
        return True
    if not force and os.path.isdir(dest_dir_path):
        if verbose:
            if input("\tThe destination folder already exists. Overwrite it? >> [Y/N] ").lower()!="y":
                return False
        else:
            return False
    try:
        return shutil.copytree(src_dir_path,dest_dir_path,dirs_exist_ok=True)==dest_dir_path
    except (FileNotFoundError,OSError):
        return False
