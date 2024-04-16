#!/usr/bin/env python3

"""Constants for the game creatures_world1."""

from collections import OrderedDict
import sys
sys.dont_write_bytecode=True

battle_moves_list=["move","struggle","move_1","move_2","move_3","move_4"]+[f"atk_{i:02d}" for i in range(1,9)]
field_moves_list=["torch","debush","swim","teleport","boulders"]
moves_list=battle_moves_list+field_moves_list
moves_ids_dict={k:i for i,k in enumerate(moves_list)}

creatures_data_list=OrderedDict([
    ("creature",    [0,300,[0xFF,0,""]]),
    ("starter_1",   [0,318,[16,1,"starter_2"]]),
    ("starter_2",   [0,405,[32,1,"starter_3"]]),
    ("starter_3",   [0,525,[0xFF,0,""]]),
    ("bird_1",      [0,300,[0xFF,0,"bird_2"]]),
    ("bird_2",      [0,370,[32,1,"bird_3"]]),
    ("bird_3",      [0,460,[0xFF,0,""]]),
    ("birb_E",      [0,600,[0xFF,0,""]]),
    ("birb_F",      [0,600,[0xFF,0,""]]),
    ("birb_I",      [0,600,[0xFF,0,""]]),
    ("monster_P",   [0,700,[0xFF,0,""]]),
])

for i,(k,v) in enumerate(creatures_data_list.items()):
    creatures_data_list[k][0]=i
creatures_ids_dict={k:i for i,(k,_) in enumerate(creatures_data_list.items())}
creatures_names=[k for k,_ in creatures_data_list.items()]
creatures_bst=[v[1] for _,v in creatures_data_list.items()]
creatures_evolution_data=[[v[2][0],v[2][1],0 if len(v[2])<2 or len(v[2][2])<1 else creatures_ids_dict[v[2][2]]] for _,v in creatures_data_list.items()]

generic_items_list=["item",
    "hp_cure_1","hp_cure_2","hp_cure_3","hp_cure_4",
    "pp_cure_1",
]
generic_learn_move_items_dict=OrderedDict([(f"l_atk_{i:02d}",f"atk_{i:02d}") for i in range(1,9)])
key_learn_move_items_dict=OrderedDict([
    ("powerup_torch","torch"),
    ("powerup_debush","debush"),
    ("powerup_swim","swim"),
    ("powerup_teleport","teleport"),
    ("powerup_boulders","boulders"),
])

generic_learn_move_items_list=[k for k,_ in generic_learn_move_items_dict.items()]
key_learn_move_items_list=[k for k,_ in key_learn_move_items_dict.items()]
learn_move_items_dict=OrderedDict()
for k,v in generic_learn_move_items_dict.items():
    generic_learn_move_items_dict[k]=moves_ids_dict.get(v,0)
    learn_move_items_dict[k]=generic_learn_move_items_dict[k]
for k,v in key_learn_move_items_dict.items():
    key_learn_move_items_dict[k]=moves_ids_dict.get(v,0)
    learn_move_items_dict[k]=key_learn_move_items_dict[k]

key_generic_items_list=["key_shop_pack","key_world_map","key_odd_stone","key_ship_ticket",
    "key_bicycle_coupon","key_bicycle","key_elevator_key","key_ghost_radar",
    "key_ghost_doll","key_water_bottle","key_megaphone","key_vulcanic_club_key",
]

consumable_items_list=generic_items_list+generic_learn_move_items_list
learn_move_items_list=generic_learn_move_items_list+key_learn_move_items_list
key_items_list=key_learn_move_items_list+key_generic_items_list
items_list=consumable_items_list+key_items_list
items_ids_dict={k:i for i,k in enumerate(items_list)}
