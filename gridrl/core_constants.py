#!/usr/bin/env python3

"""Game integer and file_path constants."""

from enum import IntEnum
from collections import OrderedDict
import sys
sys.dont_write_bytecode=True

MAP_DATA_FILENAME="map_data.json.gz"
MAP_METADATA_FILENAME="map_metadata.json"
TELEPORT_DATA_FILENAME="teleport_data.json"
EVENTS_FILENAME="events.json"
SCRIPTS_FILENAME="scripts.json"
NPCS_FILENAME="npcs.json"
POWERUPS_FILENAME="powerups.json"
MAP_PNG_FILENAME="map.png"
SPRITES_PNG_FILENAME="sprites.png"

### PLACEHOLDER COLOR ON RIGHT-MOST ELEMENTS
tile_id_colors_dict=OrderedDict([
### GENERIC TILES
    ("no_map_tile_id",[0x00,0x00,0x00]),
    ("unwalkable_tile_id",[0x3F,0x3F,0x3F]),
    ("walk_tile_id",[0xDF,0xEF,0xDF]),
    ("warp_tile_id",[0x7F,0x3F,0xBF]),
### NATURE
    ("forest_tile_id",[0x2F,0x5F,0x0F]),
    ("meadow_tile_id",[0xAF,0xFF,0x6F]),
        ("flower_tile_id",[0xFF,0x9F,0x5F]),
    ("puddle_tile_id",[0xAF,0xDF,0xFF]),
    ("grass_tile_id",[0x7F,0xBF,0x7F]),
    ("tall_grass_tile_id",[0x5F,0xAF,0x5F]),
        ("plant_tile_id",[0xFF,0xFF,0xFF]),
        ("tree_tile_id",[0xFF,0xFF,0xFF]),
    ("bush_tile_id",[0x3F,0xBF,0x7F]),
        ("soft_soil_tile_id",[0xFF,0xFF,0xFF]),
### INDOOR
    ("indoor_wall_tile_id",[0x3F,0x3F,0x3F]),
        ("indoor_furnitures_tile_id",[0xFF,0xFF,0xFF]),
        ("indoor_pc_tile_id",[0xFF,0xFF,0xFF]),
    ("indoor_floor_tile_id",[0xDF,0xEF,0xDF]),
### CITY
    ("building_tile_id",[0x3F,0x3F,0x3F]),
    ("street_tile_id",[0xDF,0xEF,0xDF]),
    ("sign_tile_id",[0xDF,0xFF,0xDF]),
### LEDGES
    ("ledge_down_tile_id",[0xAF,0x7F,0x2F]),
    ("ledge_left_tile_id",[0xBF,0x9F,0x3F]),
    ("ledge_right_tile_id",[0xAF,0x8F,0x2F]),
    ("ledge_up_tile_id",[0xBF,0x8F,0x3F]),
### SAND
    ("dune_tile_id",[0x7F,0x7F,0x8F]),
    ("sand_tile_id",[0xDF,0xDF,0xBF]),
        ("deep_sand_tile_id",[0xFF,0xFF,0xFF]),
        ("slippery_sand_tile_id",[0xFF,0xFF,0xFF]),
### WATER
    ("cliff_tile_id",[0x0F,0x5F,0xAF]),
    ("water_tile_id",[0x7F,0xBF,0xFF]),
    ("waterfall_tile_id",[0x3F,0x9F,0xEF]),
    ("deep_water_tile_id",[0x0,0x7F,0xDF]),
    ("rough_sea_tile_id",[0x4F,0xAF,0xBF]),
    ("whirlpool_tile_id",[0xBF,0xDF,0xFF]),
    ("water_current_down_tile_id",[0x6F,0x9F,0xCF]),
    ("water_current_left_tile_id",[0x4F,0x9F,0xCF]),
    ("water_current_right_tile_id",[0x4F,0x7F,0xCF]),
    ("water_current_up_tile_id",[0x6F,0x7F,0xCF]),
### UNDERWATER
        ("underwater_rock_tile_id",[0xFF,0xFF,0xFF]),
        ("underwater_tile_id",[0xFF,0xFF,0xFF]),
        ("underwater_algae_tile_id",[0xFF,0xFF,0xFF]),
### MOUNTAIN
    ("hard_mountain_tile_id",[0x5F,0x5F,0x5F]),
    ("snow_mountain_tile_id",[0xCF,0xDF,0xDF]),
    ("ground_tile_id",[0xBF,0xBF,0xBF]),
    ("snow_tile_id",[0xEF,0xFF,0xEF]),
    ("deep_snow_tile_id",[0xFF,0xFF,0xFF]),
        ("ice_tile_id",[0xAF,0xCF,0xCF]),
        ("breakable_rock_tile_id",[0xFF,0xFF,0xFF]),
        ("breakable_frozen_rock_tile_id",[0xFF,0xFF,0xFF]),
        ("mountain_climb_tile_id",[0xFF,0xFF,0xFF]),
### CAVE
    ("cave_tile_id",[0xAF,0xAF,0xAF]),
        ("breakable_wall_tile_id",[0xFF,0xFF,0xFF]),
### VULCANO
        ("lava_tile_id",[0xFF,0xFF,0xFF]),
### SWAMP
        ("marsh_tile_id",[0xFF,0xFF,0xFF]),
### INDOOR PUZZLES
        ("indoor_puzzle_warp_tile_id",[0x7F,0x3F,0xBF]),
        ("indoor_puzzle_platform_reset_tile_id",[0xFF,0xFF,0xFF]),
        ("indoor_puzzle_platform_down_tile_id",[0xFF,0xFF,0xFF]),
        ("indoor_puzzle_platform_left_tile_id",[0xFF,0xFF,0xFF]),
        ("indoor_puzzle_platform_right_tile_id",[0xFF,0xFF,0xFF]),
        ("indoor_puzzle_platform_up_tile_id",[0xFF,0xFF,0xFF]),
### HOLES AND AIR
        ("hole_tile_id",[0xFF,0xFF,0xFF]),
        ("cracked_hole_tile_id",[0xFF,0xFF,0xFF]),
        ("air_tile_id",[0xFF,0xFF,0xFF]),
### SCRIPTING
    ("active_script_tile_id",[0xBF,0x3F,0x7F]),
    ("old_script_tile_id",[0xBF,0x7F,0x7F]),
    ("active_reward_tile_id",[0xFF,0xFF,0x3F]),
    ("old_reward_tile_id",[0xBF,0xBF,0x7F]),
### NPC
    ("player_down_tile_id",[0xDF,0x8F,0x8F]),
    ("player_left_tile_id",[0xDF,0x7F,0x8F]),
    ("player_right_tile_id",[0xDF,0x7F,0x7F]),
    ("player_up_tile_id",[0xDF,0x8F,0x7F]),
    ("npc_down_tile_id",[0xFF,0xBF,0xDF]),
    ("npc_left_tile_id",[0xFF,0xAF,0xDF]),
    ("npc_right_tile_id",[0xFF,0xAF,0xCF]),
    ("npc_up_tile_id",[0xFF,0xBF,0xCF]),
    ("item_down_tile_id",[0x7F,0x5F,0xEF]),
    ("item_left_tile_id",[0x7F,0x4F,0xEF]),
    ("item_right_tile_id",[0x7F,0x4F,0xDF]),
    ("item_up_tile_id",[0x7F,0x5F,0xDF]),
### MENU
    ("generic_menu_tile_id",[0xFF,0xFF,0xFF]),
    ("generic_cursor_tile_id",[0x00,0x00,0x00]),
    ("generic_text_tile_id",[0x00,0x00,0x00]),
])


shrinked_characters_list=["E","A","T","O","I","N","S","H","R","U"]+["D","L"]
tiles_count_without_characters_encoding=len(tile_id_colors_dict)
for _ in range(0xFF-len(tile_id_colors_dict)-len(shrinked_characters_list)**2,-1,-1):
    tile_id_colors_dict[f"unused_tile{_:d}_id"]=[0x00,0x00,0x00]
for _ in range(len(shrinked_characters_list)**2-1,-1,-1):
    tile_id_colors_dict[f"enc_character{_:d}_tile_id"]=[_+0x01,_+0x01,_+0x01]

TILES=IntEnum("TILES",list(tile_id_colors_dict.keys()),start=0)

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
count_tiles_ids=len(tile_id_colors_dict)

special_walkable_tiles_ids=[
    warp_tile_id,active_script_tile_id,
    old_script_tile_id,old_reward_tile_id,
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
tile_id_colors_list=list(tile_id_colors_dict.values())
