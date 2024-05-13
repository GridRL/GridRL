#!/usr/bin/env python3

"""Game integer and file_path constants."""

from enum import IntEnum
from collections import OrderedDict
import sys
sys.dont_write_bytecode=True

#__all__=[]

MAP_DATA_FILENAME="map_data.json.gz"
MAP_METADATA_FILENAME="map_metadata.json"
TELEPORT_DATA_FILENAME="teleport_data.json"
EVENTS_FILENAME="events.json"
SCRIPTS_FILENAME="scripts.json"
NPCS_FILENAME="npcs.json"
POWERUPS_FILENAME="powerups.json"
MAP_PNG_FILENAME="map.png"
SPRITES_PNG_FILENAME="sprites.png"

### (TILE_KEY,([RGB],COMPRESSION_IMPORTANCE,PATHFINDER_WEIGHT))
tiles_data_dict=OrderedDict([
### GENERIC TILES
    ("no_map_tile_id",(         [0x00,0x00,0x00],0,0x00,0x00)),
    ("unwalkable_tile_id",(     [0x3F,0x3F,0x3F],1,0x01,0x00)),
    ("walk_tile_id",(           [0xDF,0xEF,0xDF],2,0x11,0x01)),
    ("warp_tile_id",(           [0x7F,0x3F,0xBF],3,0x71,0xFF)),
### NATURE
    ("forest_tile_id",(         [0x2F,0x5F,0x0F],1,0x01,0x00)),
    ("meadow_tile_id",(         [0xAF,0xFF,0x6F],2,0x01,0xFF)),
        ("flower_tile_id",(     [0xFF,0x9F,0x5F],2,0x01,0xFF)),
    ("puddle_tile_id",(         [0xAF,0xDF,0xFF],2,0x01,0xFF)),
    ("grass_tile_id",(          [0x7F,0xBF,0x7F],4,0x15,0x06)),
    ("tall_grass_tile_id",(     [0x5F,0xAF,0x5F],2,0x01,0xFF)),
        ("plant_tile_id",(      [0xFF,0xFF,0xFF],2,0x01,0xFF)),
        ("tree_tile_id",(       [0xFF,0xFF,0xFF],1,0x01,0x00)),
    ("bush_tile_id",(           [0x3F,0xBF,0x7F],6,0x41,0xFF)),
        ("soft_soil_tile_id",(  [0xFF,0xFF,0xFF],2,0x01,0xFF)),
### INDOOR
    ("indoor_wall_tile_id",(    [0x3F,0x3F,0x3F],1,0x01,0x00)),
        ("indoor_furnitures_tile_id",([0xFF,0xFF,0xFF],1,0x01,0x00)),
        ("indoor_pc_tile_id",(  [0xFF,0xFF,0xFF],1,0x01,0xFF)),
    ("indoor_floor_tile_id",(   [0xDF,0xEF,0xDF],1,0x01,0xFF)),
### CITY
    ("building_tile_id",(       [0x3F,0x3F,0x3F],1,0x01,0x00)),
    ("street_tile_id",(         [0xDF,0xEF,0xDF],2,0x01,0xFF)),
    ("sign_tile_id",(           [0xDF,0xFF,0xDF],2,0x01,0xFF)),
### LEDGES
    ("ledge_down_tile_id",(     [0xAF,0x7F,0x2F],2,0x24,0xFF)),
    ("ledge_left_tile_id",(     [0xBF,0x9F,0x3F],2,0x23,0xFF)),
    ("ledge_right_tile_id",(    [0xAF,0x8F,0x2F],2,0x22,0xFF)),
    ("ledge_up_tile_id",(       [0xBF,0x8F,0x3F],2,0x21,0xFF)),
### SAND
    ("dune_tile_id",(           [0x7F,0x7F,0x8F],1,0x01,0x00)),
    ("sand_tile_id",(           [0xDF,0xDF,0xBF],2,0x01,0xFF)),
        ("deep_sand_tile_id",(  [0xFF,0xFF,0xFF],2,0x01,0xFF)),
        ("slippery_sand_tile_id",([0xFF,0xFF,0xFF],2,0x01,0xFF)),
### WATER
    ("cliff_tile_id",(          [0x0F,0x5F,0xAF],1,0x01,0x00)),
    ("water_tile_id",(          [0x7F,0xBF,0xFF],5,0x31,0xFF)),
    ("waterfall_tile_id",(      [0x3F,0x9F,0xEF],2,0x01,0xFF)),
    ("deep_water_tile_id",(     [0x0,0x7F,0xDF],2,0x01,0xFF)),
    ("rough_sea_tile_id",(      [0x4F,0xAF,0xBF],2,0x01,0xFF)),
    ("whirlpool_tile_id",(      [0xBF,0xDF,0xFF],2,0x01,0xFF)),
    ("water_current_down_tile_id",([0x6F,0x9F,0xCF],2,0x01,0xFF)),
    ("water_current_left_tile_id",([0x4F,0x9F,0xCF],2,0x01,0xFF)),
    ("water_current_right_tile_id",([0x4F,0x7F,0xCF],2,0x01,0xFF)),
    ("water_current_up_tile_id",([0x6F,0x7F,0xCF],2,0x01,0xFF)),
### UNDERWATER
        ("underwater_rock_tile_id",([0xFF,0xFF,0xFF],2,0x01,0xFF)),
        ("underwater_tile_id",( [0xFF,0xFF,0xFF],2,0x01,0xFF)),
        ("underwater_algae_tile_id",([0xFF,0xFF,0xFF],2,0x01,0xFF)),
### MOUNTAIN
    ("hard_mountain_tile_id",(  [0x5F,0x5F,0x5F],1,0x01,0x00)),
    ("snow_mountain_tile_id",(  [0xCF,0xDF,0xDF],1,0x01,0x00)),
    ("ground_tile_id",(         [0xBF,0xBF,0xBF],2,0x01,0xFF)),
    ("snow_tile_id",(           [0xEF,0xFF,0xEF],2,0x01,0xFF)),
    ("deep_snow_tile_id",(      [0xFF,0xFF,0xFF],2,0x01,0xFF)),
        ("ice_tile_id",(        [0xAF,0xCF,0xCF],2,0x01,0xFF)),
        ("breakable_rock_tile_id",([0xFF,0xFF,0xFF],2,0x01,0xFF)),
        ("breakable_frozen_rock_tile_id",([0xFF,0xFF,0xFF],2,0x01,0xFF)),
        ("mountain_climb_tile_id",([0xFF,0xFF,0xFF],2,0x01,0xFF)),
### CAVE
    ("cave_tile_id",(           [0xAF,0xAF,0xAF],2,0x01,0xFF)),
        ("breakable_wall_tile_id",([0xFF,0xFF,0xFF],2,0x01,0xFF)),
### VULCANO
        ("lava_tile_id",(       [0xFF,0xFF,0xFF],1,0x01,0x00)),
### SWAMP
        ("marsh_tile_id",(      [0xFF,0xFF,0xFF],2,0x01,0xFF)),
### INDOOR PUZZLES
    ("indoor_puzzle_warp_tile_id",([0x7F,0x3F,0xBF],3,0x01,0xFF)),
    ("indoor_puzzle_platform_reset_tile_id",([0x9F,0x9F,0x9F],2,0x01,0xFF)),
    ("indoor_puzzle_platform_down_tile_id",([0x9F,0x8F,0x8F],2,0x01,0xFF)),
    ("indoor_puzzle_platform_left_tile_id",([0x9F,0x7F,0x8F],2,0x01,0xFF)),
    ("indoor_puzzle_platform_right_tile_id",([0x9F,0x7F,0x7F],2,0x01,0xFF)),
    ("indoor_puzzle_platform_up_tile_id",([0x9F,0x8F,0x7F],2,0x01,0xFF)),
### HOLES AND AIR
        ("hole_tile_id",(       [0xFF,0xFF,0xFF],2,0x01,0xFF)),
        ("cracked_hole_tile_id",([0xFF,0xFF,0xFF],2,0x01,0xFF)),
        ("air_tile_id",(        [0xFF,0xFF,0xFF],1,0x01,0x00)),
### SCRIPTING
    ("active_script_tile_id",(  [0xBF,0x3F,0x7F],2,0x61,0x50)),
    ("old_script_tile_id",(     [0xBF,0x7F,0x7F],2,0x51,0x04)),
    ("active_reward_tile_id",(  [0xFF,0xFF,0x3F],2,0x81,0x19)),
    ("old_reward_tile_id",(     [0xBF,0xBF,0x7F],2,0x65,0xFE)),
### NPC
    ("player_down_tile_id",(    [0xDF,0x8F,0x8F],7,0x00,0x00)),
    ("player_left_tile_id",(    [0xDF,0x7F,0x8F],7,0x00,0x00)),
    ("player_right_tile_id",(   [0xDF,0x7F,0x7F],7,0x00,0x00)),
    ("player_up_tile_id",(      [0xDF,0x8F,0x7F],7,0x00,0x00)),
    ("npc_down_tile_id",(       [0xFF,0xBF,0xDF],8,0x00,0x00)),
    ("npc_left_tile_id",(       [0xFF,0xAF,0xDF],8,0x00,0x00)),
    ("npc_right_tile_id",(      [0xFF,0xAF,0xCF],8,0x00,0x00)),
    ("npc_up_tile_id",(         [0xFF,0xBF,0xCF],8,0x00,0x00)),
    ("item_down_tile_id",(      [0x7F,0x5F,0xEF],9,0x00,0x00)),
    ("item_left_tile_id",(      [0x7F,0x4F,0xEF],9,0x00,0x00)),
    ("item_right_tile_id",(     [0x7F,0x4F,0xDF],9,0x00,0x00)),
    ("item_up_tile_id",(        [0x7F,0x5F,0xDF],9,0x00,0x00)),
### MENU
    ("generic_menu_tile_id",(   [0xFF,0xFF,0xFF],10,0x00,0x00)),
    ("generic_cursor_tile_id",( [0x00,0x00,0x00],10,0x00,0x00)),
    ("generic_text_tile_id",(   [0x00,0x00,0x00],10,0x00,0x00)),
])

tiles_id_colors_dict=OrderedDict([(k,v[0]) for k,v in tiles_data_dict.items()])
tiles_shrinked_id_list=[v[1] for _,v in tiles_data_dict.items()]
tiles_compression_importances_list=[v[2] for _,v in tiles_data_dict.items()]
tiles_pathfinder_weights_list=[v[3] for _,v in tiles_data_dict.items()]

shrinked_characters_list=["E","A","T","O","I","N","S","H","R","U"]+["D","L"]
tiles_count_without_characters_encoding=len(tiles_id_colors_dict)
for _ in range(0xFF-len(tiles_id_colors_dict)-len(shrinked_characters_list)**2,-1,-1):
    tiles_id_colors_dict[f"unused_tile{_:d}_id"]=[0x00,0x00,0x00]
for _ in range(len(shrinked_characters_list)**2-1,-1,-1):
    tiles_id_colors_dict[f"enc_character{_:d}_tile_id"]=[_+0x01,_+0x01,_+0x01]

TILES=IntEnum("TILES",list(tiles_id_colors_dict.keys()),start=0)

(no_map_tile_id,
unwalkable_tile_id,
walk_tile_id,
warp_tile_id,
forest_tile_id,
meadow_tile_id,
flower_tile_id,
puddle_tile_id,
grass_tile_id,
tall_grass_tile_id,
plant_tile_id,
tree_tile_id,
bush_tile_id,
soft_soil_tile_id,
indoor_wall_tile_id,
indoor_furnitures_tile_id,
indoor_pc_tile_id,
indoor_floor_tile_id,
building_tile_id,
street_tile_id,
sign_tile_id,
ledge_down_tile_id,
ledge_left_tile_id,
ledge_right_tile_id,
ledge_up_tile_id,
dune_tile_id,
sand_tile_id,
deep_sand_tile_id,
slippery_sand_tile_id,
cliff_tile_id,
water_tile_id,
waterfall_tile_id,
deep_water_tile_id,
rough_sea_tile_id,
whirlpool_tile_id,
water_current_down_tile_id,
water_current_left_tile_id,
water_current_right_tile_id,
water_current_up_tile_id,
underwater_rock_tile_id,
underwater_tile_id,
underwater_algae_tile_id,
hard_mountain_tile_id,
snow_mountain_tile_id,
ground_tile_id,
snow_tile_id,
deep_snow_tile_id,
ice_tile_id,
breakable_rock_tile_id,
breakable_frozen_rock_tile_id,
mountain_climb_tile_id,
cave_tile_id,
breakable_wall_tile_id,
lava_tile_id,
marsh_tile_id,
indoor_puzzle_warp_tile_id,
indoor_puzzle_platform_reset_tile_id,
indoor_puzzle_platform_down_tile_id,
indoor_puzzle_platform_left_tile_id,
indoor_puzzle_platform_right_tile_id,
indoor_puzzle_platform_up_tile_id,
hole_tile_id,
cracked_hole_tile_id,
air_tile_id,
active_script_tile_id,
old_script_tile_id,
active_reward_tile_id,
old_reward_tile_id,
player_down_tile_id,
player_left_tile_id,
player_right_tile_id,
player_up_tile_id,
npc_down_tile_id,
npc_left_tile_id,
npc_right_tile_id,
npc_up_tile_id,
item_down_tile_id,
item_left_tile_id,
item_right_tile_id,
item_up_tile_id,
generic_menu_tile_id,
generic_cursor_tile_id,
generic_text_tile_id,
count_non_characters_ids)=list(range(tiles_count_without_characters_encoding+1))
count_tiles_ids=len(tiles_id_colors_dict)

special_walkable_tiles_ids=[
    warp_tile_id,active_script_tile_id,
    old_script_tile_id,#old_reward_tile_id,
    indoor_puzzle_warp_tile_id,
    indoor_puzzle_platform_reset_tile_id,
    indoor_puzzle_platform_down_tile_id,
    indoor_puzzle_platform_left_tile_id,
    indoor_puzzle_platform_right_tile_id,
    indoor_puzzle_platform_up_tile_id,
    water_current_down_tile_id,water_current_left_tile_id,
    water_current_right_tile_id,water_current_up_tile_id,
]
walkable_tiles_ids=[
    walk_tile_id,
    meadow_tile_id,
    flower_tile_id,
    puddle_tile_id,
    grass_tile_id,
    tall_grass_tile_id,
    indoor_floor_tile_id,
    street_tile_id,
    ledge_down_tile_id,ledge_left_tile_id,ledge_right_tile_id,ledge_up_tile_id,
    sand_tile_id,
    underwater_tile_id,
    underwater_algae_tile_id,
    ground_tile_id,
    snow_tile_id,
    deep_snow_tile_id,
    ice_tile_id,
    cave_tile_id,
]+special_walkable_tiles_ids
swimmable_tiles_ids=[
    water_tile_id,deep_water_tile_id,
]+special_walkable_tiles_ids
tile_id_colors_list=list(tiles_id_colors_dict.values())
