#!/usr/bin/env python3

"""Images utility functions."""

from typing import Union
import sys
#import os
from io import BytesIO
import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image,ImageFont,ImageDraw
import matplotlib.pyplot as plt
#import matplotlib.font_manager as mfontm
sys.dont_write_bytecode=True

__all__=["fix_contiguous_array","tile_split","tile_rebuild","fill_pad_image",
    "rgb_to_grayscale","map_matrix_to_image","generate_characters_tiles",
    "generate_gif_from_numpy","show_image","compute_minimap"]

def fix_contiguous_array(img:np.ndarray)->np.ndarray:
    """Fixes the ndarray contiguity."""
    return img if img.flags["C_CONTIGUOUS"] else np.ascontiguousarray(img)

def tile_split(img:np.ndarray,tile_height:int,tile_width:Union[int,None]=None,contiguous:bool=True)->np.ndarray:
    """Splits an image of shape [y,x,c], into an array of tiles [y_id,x_id,h,w,c]."""
    no_color_channels=len(img.shape)<3
    (img_height,img_width,channels)=(list(img.shape)+[1])[:3]
    if tile_width is None:
        tile_width=tile_height
    bytelength=img.nbytes//img.size
    tiles=as_strided(img,
        shape=(img_height//tile_height,img_width//tile_width,tile_height,tile_width,channels),
        strides=(img_width*tile_height*bytelength*channels,tile_width*bytelength*channels,
            img_width*bytelength*channels,bytelength*channels,bytelength)
    )
    if no_color_channels:
        tiles=tiles[...,0]
    if contiguous:
        tiles=fix_contiguous_array(tiles)
    return tiles

def tile_rebuild(tiles:np.ndarray,contiguous:bool=True)->np.ndarray:
    """Reconstructs an array of tiles with shape [y_id,x_id,h,w,c] into an image [y,x,c]."""
    img=tiles.reshape(tiles.shape[0]*tiles.shape[2],tiles.shape[1]*tiles.shape[3],*tiles.shape[4:5]).swapaxes(1,2)
    if contiguous:
        img=fix_contiguous_array(img)
    return img

def fill_pad_image(img:np.ndarray,pad_y:int,pad_x:int,reserve_first_channels:bool=False,fill_value:int=0x00)->np.ndarray:
    """Pad an image on expected y/x axis. Useful for adding an extra safety space to avoid bounding issues."""
    if pad_x>0 or pad_y>0:
        pads=[max(0,int(pad_y)),max(0,int(pad_x))]
        if reserve_first_channels:
            new_img=np.full(tuple(list(img.shape[:-2])+[s+2*p for s,p in zip(img.shape[-2:],pads)]),fill_value,dtype=img.dtype,order="C")
            new_img[...,pads[0]:img.shape[-2]+pads[0],pads[1]:img.shape[-1]+pads[1]]=img
        else:
            new_img=np.full(tuple([s+2*p for s,p in zip(img.shape[:2],pads)]+list(img.shape[2:])),fill_value,dtype=img.dtype,order="C")
            new_img[pads[0]:img.shape[0]+pads[0],pads[1]:img.shape[1]+pads[1]]=img
        return new_img
    return img

def rgb_to_grayscale(img:np.ndarray,channel_axis=-1,shrink_axis:bool=True,equal_split:bool=False,contiguous:bool=True)->np.ndarray:
    """RGB to grayscale ndarray conversion."""
    if len(img.shape)==0 or 0<=channel_axis<len(img.shape) or img.shape[channel_axis]<3:
        raise ValueError(f"Axis {channel_axis} dimension of input array must be at least 3; shape {img.shape} was found.")
    img_weights=[0.3333,0.3333,0.3334] if equal_split else [0.2989,0.587,0.114]
    new_img=np.sum([w*np.take(img,i if shrink_axis else [i],axis=channel_axis) for i,w in enumerate(img_weights)],axis=0).astype(img.dtype)
    if contiguous:
        new_img=fix_contiguous_array(new_img)
    return new_img

def map_matrix_to_image(matrix:np.ndarray,colors_list:list)->np.ndarray:
    """Transform a matrix of tile indexes into an image."""
    try:
        return np.take(colors_list,matrix[:,:,0] if len(matrix.shape)>2 else matrix,axis=0)
    except IndexError:
        img=matrix.copy() if len(matrix.shape)>2 else np.repeat(np.expand_dims(matrix,axis=-1),3,axis=2)
        for idx,c in enumerate(colors_list):
            img[img[:,:,0]==idx,:]=c
        return img

def generate_characters_tiles(height:int,width:int,downscale:int=1,fill_value=0xFF)->np.ndarray:
    """Return characters as flattened tiles."""
    characters=[" "]*0x21+[chr(k) for k in range(0x21,0x7F)]+[" "]*0x81
    txt="".join(characters)
    start_fontsize=max(1,min(height,width)//max(1,int(downscale)))
    fontsize=start_fontsize
###    system_fonts=[k.split(os.sep)[-1] for k in mfontm.findSystemFonts(fontpaths=None,fontext="ttf")]
    font=None
    allowed_fonts=["OCRAEXT.TTF","CascadiaMono.ttf","consolab.ttf","Lucida-Console.ttf","couri.ttf"]
    text_shape=[]
    for font_name in allowed_fonts:
        fontsize=start_fontsize+1
        try:
            valid_fond=False
            while fontsize>=4 and not valid_fond:
                fontsize-=1
                font=ImageFont.truetype(font_name,fontsize)
                text_shape=np.array(list(font.getbbox(txt)[2:4][::-1]),dtype=np.uint16)
                text_ratio=text_shape/start_fontsize
                text_ratio[1]/=len(characters)
                if np.max(text_ratio)<=1.:
                    valid_fond=True
            if valid_fond:
                break
        except OSError:
            pass
    if font is None:
        fontsize=start_fontsize
        text_shape=[start_fontsize,start_fontsize]
    np_image=np.full((text_shape[0],text_shape[1],3),0xFF,dtype=np.uint8,order="C")
    image=Image.fromarray(np_image)
    draw=ImageDraw.Draw(image)
    draw.text((0,0),txt,font=font,fill=(0,0,0,255))
    draw.text((0,height),"."*len(txt),font=font,fill=(0,0,0,255))
    np_image=np.asarray(image)
    np_image=np.where(np_image<0xA0,0,fill_value).astype(np.uint8)
    tiles=tile_split(np_image,text_shape[0],text_shape[1]//len(txt),contiguous=False)
    tiles=fix_contiguous_array(tiles[0,:,:height,:width//2])
    return tiles

def generate_gif_from_numpy(np_imgs:list,outfile_or_buff:Union[str,BytesIO,None]=None,return_buff:bool=True,frame_duration:int=200,loop:bool=False)->Union[bool,BytesIO]:
    """Build an image from a list of ndarrays."""
    if np_imgs is None or len(np_imgs)<1:
        return False
    frames=[]
    for img in np_imgs:
        try:
            frames.append(Image.fromarray(img))
        except (AttributeError,ValueError,OSError):
            pass
    buff=BytesIO() if outfile_or_buff is None else outfile_or_buff
    if len(frames)>0:
        frames[0].save(buff,format="GIF",optimize=True,append_images=frames,save_all=True,duration=max(8,int(frame_duration)),loop=1 if loop else 0)
    if isinstance(buff,BytesIO):
        buff.seek(0)
    return buff if outfile_or_buff is None or (return_buff and isinstance(outfile_or_buff,BytesIO)) else len(frames)>0

### ONLY FUNCTION REQUIRING MATPLOTLIB
def show_image(img:np.ndarray,title:str="Img")->None:
    """Plots the image."""
    plt.style.use("dark_background")
    (_,ax)=plt.subplots(1)
    ax.imshow(img)
    if len(title)>0:
        ax.set_title(title.replace("\t"," "))
    plt.show()

def compute_minimap(full_map:np.ndarray,tiles_importance:np.ndarray,downscale:int=3)->np.ndarray:
    """Compress a 3d map in a custom representation 4d representation."""
    if len(full_map.shape)>3:
        return full_map
    used_downscale=max(2,int(downscale))
#    padding=np.mod(used_downscale-np.mod(full_map.shape[-2:],used_downscale),used_downscale)
#    if np.max(padding)>0:
#    ### TO-ADD: PAD IMAGE IF NOT DIVISIBLE
#        print("PAD!",padding)
    tile_map_data=tile_split(full_map.reshape(-1,*full_map.shape[2:]),used_downscale)
    tile_map_data=tile_map_data.reshape(full_map.shape[0],-1,*tile_map_data.shape[1:])
    tile_shape=list(tile_map_data.shape)
    tile_map_data=fix_contiguous_array(tile_map_data.reshape(*tile_map_data.shape[:3],-1))
    tile_map_importance=np.take(tiles_importance,tile_map_data,mode="clip")
    amax=np.expand_dims(np.argmax(tile_map_importance,axis=3),axis=-1)
    minimap=np.concatenate(np.unravel_index(amax,tile_shape),axis=-1).astype(np.uint8)[...,-3:]
    minimap=fix_contiguous_array(minimap)
    minimap[...,0]=np.take_along_axis(tile_map_data,amax,axis=-1)[...,0]
    return minimap
