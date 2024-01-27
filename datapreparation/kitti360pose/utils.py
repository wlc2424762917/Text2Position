"""Utilities for data preparation.
"""

import numpy as np
# from segment_anything import sam_model_registry, SamPredictor
import random


SCENE_NAMES = [
    "2013_05_28_drive_0000_sync",
    "2013_05_28_drive_0002_sync",
    "2013_05_28_drive_0003_sync",
    "2013_05_28_drive_0004_sync",
    "2013_05_28_drive_0005_sync",
    "2013_05_28_drive_0006_sync",
    "2013_05_28_drive_0007_sync",
    "2013_05_28_drive_0009_sync",
    "2013_05_28_drive_0010_sync",
]
SCENE_NAMES_TRAIN = [
    "2013_05_28_drive_0000_sync",
    "2013_05_28_drive_0002_sync",
    "2013_05_28_drive_0004_sync",
    "2013_05_28_drive_0006_sync",
    "2013_05_28_drive_0007_sync",
]
SCENE_NAMES_VAL = [
    "2013_05_28_drive_0010_sync",
]
SCENE_NAMES_TEST = [
    "2013_05_28_drive_0003_sync",
    "2013_05_28_drive_0005_sync",
    "2013_05_28_drive_0009_sync",
]

assert len(SCENE_NAMES_TRAIN + SCENE_NAMES_VAL + SCENE_NAMES_TEST) == 9
assert len(np.unique(SCENE_NAMES_TRAIN + SCENE_NAMES_VAL + SCENE_NAMES_TEST)) == 9

SCENE_SIZES = {
    "2013_05_28_drive_0000_sync": [735, 1061, 30],
    "2013_05_28_drive_0002_sync": [952, 1313, 89],
    "2013_05_28_drive_0003_sync": [713, 922, 34],
    "2013_05_28_drive_0004_sync": [1302, 2003, 60],
    "2013_05_28_drive_0005_sync": [801, 999, 51],
    "2013_05_28_drive_0006_sync": [881, 1004, 80],
    "2013_05_28_drive_0007_sync": [3049, 1989, 52],
    "2013_05_28_drive_0009_sync": [615, 1113, 26],
    "2013_05_28_drive_0010_sync": [1560, 1445, 29],
}

CLASS_TO_INDEX = {
    "building": 0,
    "pole": 1,
    "traffic light": 2,
    "traffic sign": 3,
    "garage": 4,
    "stop": 5,
    "smallpole": 6,
    "lamp": 7,
    "trash bin": 8,
    "vending machine": 9,
    "box": 10,
    "road": 11,
    "sidewalk": 12,
    "parking": 13,
    "wall": 14,
    "fence": 15,
    "guard rail": 16,
    "bridge": 17,
    "tunnel": 18,
    "vegetation": 19,
    "terrain": 20,
    "pad": 21,
}

CLASS_TO_LABEL = {
    "building": 11,
    "pole": 17,
    "traffic light": 19,
    "traffic sign": 20,
    "garage": 34,
    "stop": 36,
    "smallpole": 37,
    "lamp": 38,
    "trash bin": 39,
    "vending machine": 40,
    "box": 41,
    "road": 7,
    "sidewalk": 8,
    "parking": 9,
    "wall": 12,
    "fence": 13,
    "guard rail": 14,
    "bridge": 15,
    "tunnel": 16,
    "vegetation": 21,
    "terrain": 22,
}

CLASS_TO_COLOR = {
    "building": (70, 70, 70),
    "pole": (153, 153, 153),
    "traffic light": (250, 170, 30),
    "traffic sign": (220, 220, 0),
    "garage": (64, 128, 128),
    "stop": (150, 120, 90),
    "smallpole": (153, 153, 153),
    "lamp": (0, 64, 64),
    "trash bin": (0, 128, 192),
    "vending machine": (128, 64, 0),
    "box": (64, 64, 128),
    "sidewalk": (244, 35, 232),
    "road": (128, 64, 128),
    "parking": (250, 170, 160),
    "wall": (102, 102, 156),
    "fence": (190, 153, 153),
    "guard rail": (180, 165, 180),
    "bridge": (150, 100, 100),
    "tunnel": (150, 120, 90),
    "vegetation": (107, 142, 35),
    "terrain": (152, 251, 152),
    "_pose": (255, 255, 255),
}

CLASS_TO_MINPOINTS = {
    "building": 250,
    "pole": 25,
    "traffic light": 25,
    "traffic sign": 25,
    "garage": 250,
    "stop": 25,
    "smallpole": 25,
    "lamp": 25,
    "trash bin": 25,
    "vending machine": 25,
    "box": 25,
    "sidewalk": 1000,
    "road": 1000,
    "parking": 1000,
    "wall": 250,
    "fence": 250,
    "guard rail": 250,
    "bridge": 1000,
    "tunnel": 1000,
    "vegetation": 250,
    "terrain": 250,
    "_pose": 25,
}

CLASS_TO_VOXELSIZE = {
    "building": 0.25,
    "pole": None,
    "traffic light": None,
    "traffic sign": None,
    "garage": 0.125,
    "stop": None,
    "smallpole": None,
    "lamp": None,
    "trash bin": None,
    "vending machine": None,
    "box": None,
    "sidewalk": 0.25,
    "road": 0.25,
    "parking": 0.25,
    "wall": 0.125,
    "fence": 0.125,
    "guard rail": 0.125,
    "bridge": 0.25,
    "tunnel": 0.25,
    "vegetation": 0.25,
    "terrain": 0.25,
    "_pose": None,
}

STUFF_CLASSES = [
    "sidewalk",
    "road",
    "parking",
    "wall",
    "fence",
    "guard rail",
    "bridge",
    "tunnel",
    "vegetation",
    "terrain",
]

LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

COLORS = (
    np.array(
        [
            [47.2579917, 49.75368454, 42.4153065],
            [136.32696657, 136.95241796, 126.02741229],
            [87.49822126, 91.69058836, 80.14558512],
            [213.91030679, 216.25033052, 207.24611073],
            [110.39218852, 112.91977458, 103.68638249],
            [27.47505158, 28.43996795, 25.16840296],
            [66.65951839, 70.22342483, 60.20395996],
            [171.00852191, 170.05737735, 155.00130334],
        ]
    )
    / 255.0
)

CLASS_NAMES_ALTERNATIVES = {
    "building": ["structure", "edifice", "construction", "complex", "facility", "block"],
    "pole": ["post", "column", "rod", "shaft", "pillar", "mast"],
    "traffic light": ["signal", "stoplight", "semaphore", "beacon", "traffic signal", "indicator"],
    "traffic sign": ["road sign", "street sign", "signpost", "warning sign", "traffic marker", "direction sign"],
    "garage": ["carport", "auto shop", "workshop", "shed", "storage", "parking lot"],
    "stop": ["halt", "pause", "standstill", "cessation", "break", "stoppage"],
    "smallpole": ["small post", "small rod", "thin pole", "short column", "slender stake", "mini pole"],
    "lamp": ["light", "lantern", "bulb", "fixture", "torch", "illuminant"],
    "trash bin": ["garbage can", "waste basket", "rubbish bin", "dustbin", "litter bin", "recycling bin"],
    "vending machine": ["dispenser", "automat", "coin machine", "kiosk", "snack machine", "drink machine"],
    "box": ["container", "crate", "case", "bin", "chest", "packet"],
    "road": ["street", "highway", "avenue", "path", "route", "lane"],
    "sidewalk": ["pavement", "footpath", "walkway", "pathway", "esplanade", "promenade"],
    "parking": ["car park", "parking lot", "garage", "parking space", "parking area", "car space"],
    "wall": ["barrier", "partition", "panel", "divider", "fence", "screen"],
    "fence": ["barrier", "railing", "hedge", "enclosure", "palisade", "picket"],
    "guard rail": ["railing", "barricade", "barrier", "handrail", "safety rail", "parapet"],
    "bridge": ["overpass", "viaduct", "footbridge", "span", "crossing", "arch"],
    "tunnel": ["underpass", "passage", "subway", "conduit", "tube", "duct"],
    "vegetation": ["plants", "greenery", "foliage", "flora", "shrubs", "grass"],
    "terrain": ["land", "ground", "landscape", "topography", "area", "region"],
    "pad": ["cushion", "mat", "pillow", "rest", "support", "buffer"],
}

COLOR_NAMES_ALTERNATIVES = {
    "dark-green": ["forest-green", "moss-green", "olive-green", "dark-olive", "pine-green", "deep-green"],
    "gray": ["slate-gray", "silver", "stone", "stone-gray", "charcoal", "gunmetal-gray"],
    "gray-green": ["sage", "olive-drab", "moss", "sea green", "feldgrau", "willow green"],
    "bright-gray": ["light-gray", "silver", "pale-gray", "platinum", "misty-gray", "ash"],
    "black": ["jet-black", "charcoal", "onyx", "obsidian", "raven", "ink-black"],
    "green": ["kelly-green", "jade", "emerald", "grass-green", "shamrock-green", "lime-green"],
    "beige": ["tan", "sand", "khaki", "ecru", "buff", "fawn"]
}

def randomize_class_and_color(class_name, color_name):
    # 从类别近义词字典中随机选择一个近义词
    if class_name in CLASS_NAMES_ALTERNATIVES:
        class_alternative = random.choice(CLASS_NAMES_ALTERNATIVES[class_name])
    else:
        class_alternative = class_name  # 如果没有找到类别，则保持不变

    # 从颜色近义词字典中随机选择一个近义词
    if color_name in COLOR_NAMES_ALTERNATIVES:
        color_alternative = random.choice(COLOR_NAMES_ALTERNATIVES[color_name])
    else:
        color_alternative = color_name  # 如果没有找到颜色，则保持不变
    # print(f"old class: {class_name}, new class: {class_alternative}")
    return class_alternative, color_alternative


# COLOR_NAMES = ['color-0', 'color-1', 'color-2', 'color-3', 'color-4', 'color-5', 'color-6', 'color-7']
"""
Note that these names are not completely precise as the fitted colors are mostly gray-scale.
However, the models just learn them as words without deeper meaning, so they don't have a more complex effect.
"""
COLOR_NAMES = ["dark-green", "gray", "gray-green", "bright-gray", "gray", "black", "green", "beige"]

from scipy.spatial.distance import cdist

dists = cdist(COLORS, COLORS)

# from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch


def initialize_sam_model(device, sam_model_type='vit_h', sam_checkpoint='/home/wanglichao/Text2Position/checkpoints/sam_vit_h_4b8939.pth'):
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor_sam = SamPredictor(sam)
    return predictor_sam


def mask2box(mask: torch.Tensor):
    row = torch.nonzero(mask.sum(axis=0))[:, 0]
    if len(row) == 0:
        return None
    x1 = row.min().item()
    x2 = row.max().item()
    col = np.nonzero(mask.sum(axis=1))[:, 0]
    y1 = col.min().item()
    y2 = col.max().item()
    return x1, y1, x2 + 1, y2 + 1


def mask2box_multi_level(mask: torch.Tensor, level, expansion_ratio):
    x1, y1, x2, y2 = mask2box(mask)
    if level == 0:
        return x1, y1, x2, y2
    shape = mask.shape
    x_exp = int(abs(x2 - x1) * expansion_ratio) * level
    y_exp = int(abs(y2 - y1) * expansion_ratio) * level
    return max(0, x1 - x_exp), max(0, y1 - y_exp), min(shape[1], x2 + x_exp), min(shape[0], y2 + y_exp)


def run_sam(image_size, num_random_rounds, num_selected_points, point_coords, predictor_sam):
    best_score = 0
    best_mask = np.zeros_like(image_size, dtype=bool)
    point_coords_new = np.zeros_like(point_coords)

    # (x, y) --> (y, x)
    point_coords_new[:, 0] = point_coords[:, 0]
    point_coords_new[:, 1] = point_coords[:, 1]

    # Get only a random subsample of them for num_random_rounds times and choose the mask with highest confidence score
    for i in range(num_random_rounds):
        np.random.shuffle(point_coords_new)
        masks, scores, logits = predictor_sam.predict(
            point_coords=point_coords_new[:num_selected_points],
            point_labels=np.ones(point_coords_new[:num_selected_points].shape[0]),
            multimask_output=False,
        )
        if scores[0] > best_score:
            best_score = scores[0]
            best_mask = masks[0]

    return best_mask