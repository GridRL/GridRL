#!/usr/bin/env python3

"""DataFrame utility functions."""

import sys
import zlib
import pandas as pd
sys.dont_write_bytecode=True

__all__=["is_dataframe","read_dataframe","write_dataframe","build_dataframe"]

def is_series(_obj:object)->bool:
    """Return if the object is a pandas Series."""
    return isinstance(_obj,pd.Series)

def is_dataframe(obj:object)->bool:
    """Return if the object is a pandas DataFrame."""
    return isinstance(obj,pd.DataFrame)

def get_empty_dataframe(columns:list=None)->pd.DataFrame:
    """Return an empty dataframe."""
    tdf=pd.DataFrame([],columns=[] if columns is None else columns)
    tdf.reset_index(drop=True,inplace=True)
    return tdf

def is_parquet_filename(filename:str)->bool:
    """Return if a filename ends with parquet extension."""
    return filename.endswith(".parquet")

def read_dataframe(filename:str,sep=";")->pd.DataFrame:
    """Read dataframe content from a file."""
    try:
        if is_parquet_filename(filename):
            tdf=pd.read_parquet(filename,engine="fastparquet")
        else:
            tdf=pd.read_csv(filename,sep=sep,escapechar='\\',compression="infer",
                on_bad_lines="skip",encoding="ISO-8859-1")
        tdf.reset_index(drop=True,inplace=True)
        return tdf
    except (ValueError,OSError,zlib.error):
        return get_empty_dataframe()

def write_dataframe(df,filename:str,sep=";")->bool:
    """Save dataframe to a file."""
    if not is_dataframe(df):
        return False
    try:
        if is_parquet_filename(filename):
            df.to_parquet(filename,engine="fastparquet",compression="gzip",index=False,
                has_nulls=False,object_encoding="utf8",append=False)
        else:
            df.to_csv(filename,sep=sep,compression="infer",index=False,
                encoding="ISO-8859-1",decimal=".")
        return True
    except (PermissionError,OSError):
        return False

def build_dataframe(_data,_columns:list=None,_indexes=None,
    _reset_index:bool=False,_types:list=None
)->pd.DataFrame:
    """Build dataframe from a list of data, with given settings."""
    if _columns is not None and isinstance(_columns,(list,tuple,pd.core.indexes.base.Index)):
        if _indexes is not None and not isinstance(_indexes,bool):
            tdf=pd.DataFrame(_data,columns=list(_columns),index=_indexes)
        else:
            tdf=pd.DataFrame(_data,columns=list(_columns))
    else:
        tdf=pd.DataFrame(_data)
    if _types is not None:
        for k,v in zip(_columns,_types):
            tdf[k]=tdf[k].astype(v)
    if _reset_index:
        tdf.reset_index(drop=True,inplace=True)
    return tdf
