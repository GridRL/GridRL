#!/usr/bin/env python3

"""Porting some functions to avoid installing extra dependencies."""

from typing import Union
import sys
import numpy as np
from numpy.lib.stride_tricks import as_strided
sys.dont_write_bytecode=True

#__all__=[]

(HAS_MODULE_SKIMAGE,HAS_MODULE_MATPLOTLIB,HAS_MODULE_KERAS)=(False,False,False)

#try:
#    raise ModuleNotFoundError
#    from matplotlib.colors import rgb_to_hsv,hsv_to_rgb
#    HAS_MODULE_MATPLOTLIB=True
#except (ModuleNotFoundError,ImportError):
#    HAS_MODULE_MATPLOTLIB=False

#try:
#    raise ModuleNotFoundError
#    from skimage.transform import downscale_local_mean
#    HAS_MODULE_SKIMAGE=True
#except (ModuleNotFoundError,ImportError):
#    HAS_MODULE_SKIMAGE=False

#try:
#    raise ModuleNotFoundError
#    from keras.utils import to_categorical
#    HAS_MODULE_KERAS=True
#except (ModuleNotFoundError,ImportError):
#    HAS_MODULE_KERAS=False

if not HAS_MODULE_MATPLOTLIB:
    def rgb_to_hsv(arr:np.ndarray)->np.ndarray:
        """matplotlib.colors - RGB to HSV ndarray conversion."""
        arr=np.asarray(arr)
        if arr.shape[-1]!=3:
            raise ValueError("Last dimension of input array must be 3; " f"shape {arr.shape} was found.")
        in_shape=arr.shape
        arr=np.array(arr,copy=False,dtype=np.promote_types(arr.dtype,np.float32),ndmin=2)
        out=np.zeros_like(arr)
        arr_max=arr.max(-1)
        ipos=arr_max>0
        delta=np.ptp(arr,-1)
        s=np.zeros_like(delta)
        s[ipos]=delta[ipos]/arr_max[ipos]
        ipos=delta>0
        for ch_idx,(ch1,ch2) in enumerate(zip([1,2,0],[2,0,1])):
            idx=(arr[...,ch_idx]==arr_max)&ipos
            out[idx,0]=2.*ch_idx+(arr[idx,ch1]-arr[idx,ch2])/delta[idx]
        out[...,0]=(out[...,0]/6.0)%1.0
        out[...,1]=s
        out[...,2]=arr_max
        return out.reshape(in_shape)

    def hsv_to_rgb(hsv:np.ndarray)->np.ndarray:
        """matplotlib.colors - HSV to RGB ndarray conversion."""
        hsv=np.asarray(hsv,order="C")
        if hsv.shape[-1]!=3:
            raise ValueError("Last dimension of input array must be 3; " f"shape {hsv.shape} was found.")
        in_shape=hsv.shape
        hsv=np.array(hsv,copy=False,dtype=np.promote_types(hsv.dtype,np.float32),ndmin=2)
        (h,s)=[hsv[...,i] for i in range(2)]
        (r,g,b)=[np.empty_like(h) for i in range(3)]
        i=(h*6.0).astype(int)
        f=(h*6.0)-i
        img_vals={"v":hsv[...,2]}
        img_vals.update({"p":img_vals["v"]*(1.0-s),"q":img_vals["v"]*(1.0-s*f),"t":img_vals["v"]*(1.0-s*(1.0-f))})
        for v_idx,v1,v2,v3 in zip([-1,1,2,3,4,5,0],["v","q","p","p","t","v","v"],["t","v","v","q","p","p","v"],["p","p","t","v","v","q","v"]):
            idx=i%6==0 if v_idx==-1 else i==v_idx
            r[idx]=img_vals[v1][idx]
            g[idx]=img_vals[v2][idx]
            b[idx]=img_vals[v3][idx]
        rgb=np.stack([r,g,b],axis=-1)
        return rgb.reshape(in_shape)

if not HAS_MODULE_SKIMAGE:
    def downscale_local_mean(image:np.ndarray,factors:int=2,cval:int=0,
        clip:bool=True,func_kwargs:Union[dict,None]=None)->np.ndarray:
        """skimage.transform - ndarray shape scaling."""
        if np.isscalar(factors):
            factors=(factors,)*image.ndim
        elif len(factors)!=image.ndim:
            raise ValueError("`factors` must be a scalar or have the same length as `image.shape`")
        if func_kwargs is None:
            func_kwargs={}
        pad_width=[]
        for i,_ in enumerate(factors):
            if factors[i]<1:
                raise ValueError("Down-sampling factors must be >= 1. Use `skimage.transform.resize` to up-sample an image.")
            after_width=factors[i]-(image.shape[i]%factors[i]) if image.shape[i]%factors[i]!=0 else 0
            pad_width.append((0,after_width))
        if np.any(np.asarray(pad_width)):
            image=np.pad(image,pad_width=pad_width,mode='constant',constant_values=cval)
        if not isinstance(factors,tuple):
            raise TypeError('factors needs to be a tuple')
        factors=np.array(factors)
        if (factors<=0).any():
            raise ValueError("'factors' elements must be strictly positive")
        if factors.size!=image.ndim:
            raise ValueError("'factors' must have the same length as 'image.shape'")
        arr_shape=np.array(image.shape)
        if (arr_shape%factors).sum()!=0:
            raise ValueError("'factors' is not compatible with 'image'")
        new_shape=tuple(arr_shape//factors)+tuple(factors)
        new_strides=tuple(image.strides*factors)+image.strides
        blocked=as_strided(image,shape=new_shape,strides=new_strides)
        return np.mean(blocked,axis=tuple(range(image.ndim,blocked.ndim)),**func_kwargs)

if not HAS_MODULE_KERAS:
    def to_categorical(x:np.ndarray,num_classes:int=0,dtype="int64")->np.ndarray:
        """keras.utils - ndarray num_classes one-hot encoding."""
        x=np.array(x,dtype=dtype)
        input_shape=x.shape
        if input_shape and input_shape[-1]==1 and len(input_shape)>1:
            input_shape=tuple(input_shape[:-1])
        x=x.reshape(-1)
        if not num_classes:
            num_classes=np.max(x)+1
        batch_size=x.shape[0]
        categorical=np.zeros((batch_size,num_classes),dtype=dtype,order="C")
        try:
            categorical[np.arange(batch_size),x]=1
        except IndexError:
            num_classes=int(np.max(x))+1
            categorical=np.zeros((batch_size,num_classes),dtype=dtype,order="C")
            categorical[np.arange(batch_size),x]=1
        output_shape=input_shape+(num_classes,)
        categorical=np.reshape(categorical,output_shape)
        return categorical
