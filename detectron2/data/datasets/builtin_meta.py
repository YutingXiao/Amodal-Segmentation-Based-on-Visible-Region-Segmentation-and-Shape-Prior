# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# All coco categories, together with their nice-looking visualization colors
# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json
COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
    {"color": [255, 255, 128], "isthing": 0, "id": 92, "name": "banner"},
    {"color": [147, 211, 203], "isthing": 0, "id": 93, "name": "blanket"},
    {"color": [150, 100, 100], "isthing": 0, "id": 95, "name": "bridge"},
    {"color": [168, 171, 172], "isthing": 0, "id": 100, "name": "cardboard"},
    {"color": [146, 112, 198], "isthing": 0, "id": 107, "name": "counter"},
    {"color": [210, 170, 100], "isthing": 0, "id": 109, "name": "curtain"},
    {"color": [92, 136, 89], "isthing": 0, "id": 112, "name": "door-stuff"},
    {"color": [218, 88, 184], "isthing": 0, "id": 118, "name": "floor-wood"},
    {"color": [241, 129, 0], "isthing": 0, "id": 119, "name": "flower"},
    {"color": [217, 17, 255], "isthing": 0, "id": 122, "name": "fruit"},
    {"color": [124, 74, 181], "isthing": 0, "id": 125, "name": "gravel"},
    {"color": [70, 70, 70], "isthing": 0, "id": 128, "name": "house"},
    {"color": [255, 228, 255], "isthing": 0, "id": 130, "name": "light"},
    {"color": [154, 208, 0], "isthing": 0, "id": 133, "name": "mirror-stuff"},
    {"color": [193, 0, 92], "isthing": 0, "id": 138, "name": "net"},
    {"color": [76, 91, 113], "isthing": 0, "id": 141, "name": "pillow"},
    {"color": [255, 180, 195], "isthing": 0, "id": 144, "name": "platform"},
    {"color": [106, 154, 176], "isthing": 0, "id": 145, "name": "playingfield"},
    {"color": [230, 150, 140], "isthing": 0, "id": 147, "name": "railroad"},
    {"color": [60, 143, 255], "isthing": 0, "id": 148, "name": "river"},
    {"color": [128, 64, 128], "isthing": 0, "id": 149, "name": "road"},
    {"color": [92, 82, 55], "isthing": 0, "id": 151, "name": "roof"},
    {"color": [254, 212, 124], "isthing": 0, "id": 154, "name": "sand"},
    {"color": [73, 77, 174], "isthing": 0, "id": 155, "name": "sea"},
    {"color": [255, 160, 98], "isthing": 0, "id": 156, "name": "shelf"},
    {"color": [255, 255, 255], "isthing": 0, "id": 159, "name": "snow"},
    {"color": [104, 84, 109], "isthing": 0, "id": 161, "name": "stairs"},
    {"color": [169, 164, 131], "isthing": 0, "id": 166, "name": "tent"},
    {"color": [225, 199, 255], "isthing": 0, "id": 168, "name": "towel"},
    {"color": [137, 54, 74], "isthing": 0, "id": 171, "name": "wall-brick"},
    {"color": [135, 158, 223], "isthing": 0, "id": 175, "name": "wall-stone"},
    {"color": [7, 246, 231], "isthing": 0, "id": 176, "name": "wall-tile"},
    {"color": [107, 255, 200], "isthing": 0, "id": 177, "name": "wall-wood"},
    {"color": [58, 41, 149], "isthing": 0, "id": 178, "name": "water-other"},
    {"color": [183, 121, 142], "isthing": 0, "id": 180, "name": "window-blind"},
    {"color": [255, 73, 97], "isthing": 0, "id": 181, "name": "window-other"},
    {"color": [107, 142, 35], "isthing": 0, "id": 184, "name": "tree-merged"},
    {"color": [190, 153, 153], "isthing": 0, "id": 185, "name": "fence-merged"},
    {"color": [146, 139, 141], "isthing": 0, "id": 186, "name": "ceiling-merged"},
    {"color": [70, 130, 180], "isthing": 0, "id": 187, "name": "sky-other-merged"},
    {"color": [134, 199, 156], "isthing": 0, "id": 188, "name": "cabinet-merged"},
    {"color": [209, 226, 140], "isthing": 0, "id": 189, "name": "table-merged"},
    {"color": [96, 36, 108], "isthing": 0, "id": 190, "name": "floor-other-merged"},
    {"color": [96, 96, 96], "isthing": 0, "id": 191, "name": "pavement-merged"},
    {"color": [64, 170, 64], "isthing": 0, "id": 192, "name": "mountain-merged"},
    {"color": [152, 251, 152], "isthing": 0, "id": 193, "name": "grass-merged"},
    {"color": [208, 229, 228], "isthing": 0, "id": 194, "name": "dirt-merged"},
    {"color": [206, 186, 171], "isthing": 0, "id": 195, "name": "paper-merged"},
    {"color": [152, 161, 64], "isthing": 0, "id": 196, "name": "food-other-merged"},
    {"color": [116, 112, 0], "isthing": 0, "id": 197, "name": "building-other-merged"},
    {"color": [0, 114, 143], "isthing": 0, "id": 198, "name": "rock-merged"},
    {"color": [102, 102, 156], "isthing": 0, "id": 199, "name": "wall-other-merged"},
    {"color": [250, 141, 255], "isthing": 0, "id": 200, "name": "rug-merged"},
]

COCOA_CLS_CATEGORIES = [
    {'supercategory': 'person', 'id': 1, 'name': 'person', 'color': [119, 11, 32]},
    {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle', 'color': [0, 0, 142]},
    {'supercategory': 'vehicle', 'id': 3, 'name': 'car', 'color': [0, 0, 230]},
    {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle', 'color': [106, 0, 228]},
    {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane', 'color': [0, 60, 100]},
    {'supercategory': 'vehicle', 'id': 6, 'name': 'bus', 'color': [0, 80, 100]},
    {'supercategory': 'vehicle', 'id': 7, 'name': 'train', 'color': [0, 0, 70]},
    {'supercategory': 'vehicle', 'id': 8, 'name': 'truck', 'color': [0, 0, 192]},
    {'supercategory': 'vehicle', 'id': 9, 'name': 'boat', 'color': [250, 170, 30]},
    {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light', 'color': [100, 170, 30]},
    {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant', 'color': [220, 220, 0]},
    {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign', 'color': [250, 0, 30]},
    {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter', 'color': [165, 42, 42]},
    {'supercategory': 'outdoor', 'id': 15, 'name': 'bench', 'color': [255, 77, 255]},
    {'supercategory': 'animal', 'id': 16, 'name': 'bird', 'color': [0, 226, 252]},
    {'supercategory': 'animal', 'id': 17, 'name': 'cat', 'color': [182, 182, 255]},
    {'supercategory': 'animal', 'id': 18, 'name': 'dog', 'color': [0, 82, 0]},
    {'supercategory': 'animal', 'id': 19, 'name': 'horse', 'color': [120, 166, 157]},
    {'supercategory': 'animal', 'id': 20, 'name': 'sheep', 'color': [110, 76, 0]},
    {'supercategory': 'animal', 'id': 21, 'name': 'cow', 'color': [174, 57, 255]},
    {'supercategory': 'animal', 'id': 22, 'name': 'elephant', 'color': [199, 100, 0]},
    {'supercategory': 'animal', 'id': 23, 'name': 'bear', 'color': [72, 0, 118]},
    {'supercategory': 'animal', 'id': 24, 'name': 'zebra', 'color': [255, 179, 240]},
    {'supercategory': 'animal', 'id': 25, 'name': 'giraffe', 'color': [0, 125, 92]},
    {'supercategory': 'accessory', 'id': 27, 'name': 'backpack', 'color': [188, 208, 182]},
    {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella', 'color': [0, 220, 176]},
    {'supercategory': 'accessory', 'id': 31, 'name': 'handbag', 'color': [133, 129, 255]},
    {'supercategory': 'accessory', 'id': 32, 'name': 'tie', 'color': [78, 180, 255]},
    {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase', 'color': [0, 228, 0]},
    {'supercategory': 'sports', 'id': 34, 'name': 'frisbee', 'color': [174, 255, 243]},
    {'supercategory': 'sports', 'id': 35, 'name': 'skis', 'color': [45, 89, 255]},
    {'supercategory': 'sports', 'id': 36, 'name': 'snowboard', 'color': [134, 134, 103]},
    {'supercategory': 'sports', 'id': 37, 'name': 'sports ball', 'color': [145, 148, 174]},
    {'supercategory': 'sports', 'id': 38, 'name': 'kite', 'color': [255, 208, 186]},
    {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat', 'color': [197, 226, 255]},
    {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove', 'color': [171, 134, 1]},
    {'supercategory': 'sports', 'id': 41, 'name': 'skateboard', 'color': [109, 63, 54]},
    {'supercategory': 'sports', 'id': 42, 'name': 'surfboard', 'color': [207, 138, 255]},
    {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket', 'color': [151, 0, 95]},
    {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle', 'color': [9, 80, 61]},
    {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass', 'color': [74, 65, 105]},
    {'supercategory': 'kitchen', 'id': 47, 'name': 'cup', 'color': [166, 196, 102]},
    {'supercategory': 'kitchen', 'id': 48, 'name': 'fork', 'color': [208, 195, 210]},
    {'supercategory': 'kitchen', 'id': 49, 'name': 'knife', 'color': [255, 109, 65]},
    {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon', 'color': [0, 143, 149]},
    {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl', 'color': [179, 0, 194]},
    {'supercategory': 'food', 'id': 52, 'name': 'banana', 'color': [209, 99, 106]},
    {'supercategory': 'food', 'id': 53, 'name': 'apple', 'color': [5, 121, 0]},
    {'supercategory': 'food', 'id': 54, 'name': 'sandwich', 'color': [227, 255, 205]},
    {'supercategory': 'food', 'id': 55, 'name': 'orange', 'color': [147, 186, 208]},
    {'supercategory': 'food', 'id': 56, 'name': 'broccoli', 'color': [153, 69, 1]},
    {'supercategory': 'food', 'id': 57, 'name': 'carrot', 'color': [3, 95, 161]},
    {'supercategory': 'food', 'id': 58, 'name': 'hot dog', 'color': [163, 255, 0]},
    {'supercategory': 'food', 'id': 59, 'name': 'pizza', 'color': [119, 0, 170]},
    {'supercategory': 'food', 'id': 60, 'name': 'donut', 'color': [0, 182, 199]},
    {'supercategory': 'food', 'id': 61, 'name': 'cake', 'color': [0, 165, 120]},
    {'supercategory': 'furniture', 'id': 62, 'name': 'chair', 'color': [183, 130, 88]},
    {'supercategory': 'furniture', 'id': 63, 'name': 'couch', 'color': [95, 32, 0]},
    {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant', 'color': [130, 114, 135]},
    {'supercategory': 'furniture', 'id': 65, 'name': 'bed', 'color': [110, 129, 133]},
    {'supercategory': 'furniture', 'id': 67, 'name': 'dining table', 'color': [219, 142, 185]},
    {'supercategory': 'furniture', 'id': 70, 'name': 'toilet', 'color': [65, 70, 15]},
    {'supercategory': 'electronic', 'id': 72, 'name': 'tv', 'color': [59, 105, 106]},
    {'supercategory': 'electronic', 'id': 73, 'name': 'laptop', 'color': [142, 108, 45]},
    {'supercategory': 'electronic', 'id': 74, 'name': 'mouse', 'color': [196, 172, 0]},
    {'supercategory': 'electronic', 'id': 75, 'name': 'remote', 'color': [95, 54, 80]},
    {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard', 'color': [128, 76, 255]},
    {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone', 'color': [201, 57, 1]},
    {'supercategory': 'appliance', 'id': 78, 'name': 'microwave', 'color': [246, 0, 122]},
    {'supercategory': 'appliance', 'id': 79, 'name': 'oven', 'color': [191, 162, 208]},
    {'supercategory': 'appliance', 'id': 80, 'name': 'toaster', 'color': [255, 255, 128]},
    {'supercategory': 'appliance', 'id': 81, 'name': 'sink', 'color': [147, 211, 203]},
    {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator', 'color': [150, 100, 100]},
    {'supercategory': 'indoor', 'id': 84, 'name': 'book', 'color': [146, 112, 198]},
    {'supercategory': 'indoor', 'id': 85, 'name': 'clock', 'color': [210, 170, 100]},
    {'supercategory': 'indoor', 'id': 86, 'name': 'vase', 'color': [92, 136, 89]},
    {'supercategory': 'indoor', 'id': 87, 'name': 'scissors', 'color': [218, 88, 184]},
    {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear', 'color': [241, 129, 0]},
    {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier', 'color': [217, 17, 255]},
    {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush', 'color': [124, 74, 181]},
]

# fmt: off
COCO_PERSON_KEYPOINT_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)
# fmt: on

# Pairs of keypoints that should be exchanged under horizontal flipping
COCO_PERSON_KEYPOINT_FLIP_MAP = (
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)

# rules for pairs of keypoints to draw a line between, and the line color to use.
KEYPOINT_CONNECTION_RULES = [
    # face
    ("left_ear", "left_eye", (102, 204, 255)),
    ("right_ear", "right_eye", (51, 153, 255)),
    ("left_eye", "nose", (102, 0, 204)),
    ("nose", "right_eye", (51, 102, 255)),
    # upper-body
    ("left_shoulder", "right_shoulder", (255, 128, 0)),
    ("left_shoulder", "left_elbow", (153, 255, 204)),
    ("right_shoulder", "right_elbow", (128, 229, 255)),
    ("left_elbow", "left_wrist", (153, 255, 153)),
    ("right_elbow", "right_wrist", (102, 255, 224)),
    # lower-body
    ("left_hip", "right_hip", (255, 102, 0)),
    ("left_hip", "left_knee", (255, 255, 77)),
    ("right_hip", "right_knee", (153, 255, 204)),
    ("left_knee", "left_ankle", (191, 255, 128)),
    ("right_knee", "right_ankle", (255, 195, 77)),
]

D2SA_CATEGORIES = [
     {'color': [220, 20, 60], 'isthing': 1, 'id': 1, 'name': '1'},
     {'color': [119, 11, 32], 'isthing': 1, 'id': 2, 'name': '2'},
     {'color': [0, 0, 142], 'isthing': 1, 'id': 3, 'name': '3'},
     {'color': [0, 0, 230], 'isthing': 1, 'id': 4, 'name': '4'},
     {'color': [106, 0, 228], 'isthing': 1, 'id': 5, 'name': '5'},
     {'color': [0, 60, 100], 'isthing': 1, 'id': 6, 'name': '6'},
     {'color': [0, 80, 100], 'isthing': 1, 'id': 7, 'name': '7'},
     {'color': [0, 0, 70], 'isthing': 1, 'id': 8, 'name': '8'},
     {'color': [0, 0, 192], 'isthing': 1, 'id': 9, 'name': '9'},
     {'color': [250, 170, 30], 'isthing': 1, 'id': 10, 'name': '10'},
     {'color': [100, 170, 30], 'isthing': 1, 'id': 11, 'name': '11'},
     {"color": [147, 186, 208], "isthing": 1, "id": 12, "name": "12"},
     {'color': [220, 220, 0], 'isthing': 1, 'id': 13, 'name': '13'},
     {'color': [175, 116, 175], 'isthing': 1, 'id': 14, 'name': '14'},
     {'color': [250, 0, 30], 'isthing': 1, 'id': 15, 'name': '15'},
     {'color': [165, 42, 42], 'isthing': 1, 'id': 16, 'name': '16'},
     {'color': [255, 77, 255], 'isthing': 1, 'id': 17, 'name': '17'},
     {'color': [0, 226, 252], 'isthing': 1, 'id': 18, 'name': '18'},
     {'color': [182, 182, 255], 'isthing': 1, 'id': 19, 'name': '19'},
     {'color': [0, 82, 0], 'isthing': 1, 'id': 20, 'name': '20'},
     {'color': [120, 166, 157], 'isthing': 1, 'id': 21, 'name': '21'},
     {'color': [110, 76, 0], 'isthing': 1, 'id': 22, 'name': '22'},
     {'color': [174, 57, 255], 'isthing': 1, 'id': 23, 'name': '23'},
     {'color': [199, 100, 0], 'isthing': 1, 'id': 24, 'name': '24'},
     {'color': [72, 0, 118], 'isthing': 1, 'id': 25, 'name': '25'},
     {"color": [153, 69, 1], "isthing": 1, "id": 26, "name": "26"},
     {'color': [255, 179, 240], 'isthing': 1, 'id': 27, 'name': '27'},
     {'color': [0, 125, 92], 'isthing': 1, 'id': 28, 'name': '28'},
     {"color": [3, 95, 161], "isthing": 1, "id": 29, "name": "29"},
     {"color": [163, 255, 0], "isthing": 1, "id": 30, "name": "30"},
     {'color': [209, 0, 151], 'isthing': 1, 'id': 31, 'name': '31'},
     {'color': [188, 208, 182], 'isthing': 1, 'id': 32, 'name': '32'},
     {'color': [0, 220, 176], 'isthing': 1, 'id': 33, 'name': '33'},
     {'color': [255, 99, 164], 'isthing': 1, 'id': 34, 'name': '34'},
     {'color': [92, 0, 73], 'isthing': 1, 'id': 35, 'name': '35'},
     {'color': [133, 129, 255], 'isthing': 1, 'id': 36, 'name': '36'},
     {'color': [78, 180, 255], 'isthing': 1, 'id': 37, 'name': '37'},
     {'color': [0, 228, 0], 'isthing': 1, 'id': 38, 'name': '38'},
     {'color': [174, 255, 243], 'isthing': 1, 'id': 39, 'name': '39'},
     {'color': [45, 89, 255], 'isthing': 1, 'id': 40, 'name': '40'},
     {'color': [134, 134, 103], 'isthing': 1, 'id': 41, 'name': '41'},
     {'color': [145, 148, 174], 'isthing': 1, 'id': 42, 'name': '42'},
     {'color': [255, 208, 186], 'isthing': 1, 'id': 43, 'name': '43'},
     {'color': [197, 226, 255], 'isthing': 1, 'id': 44, 'name': '44'},
     {"color": [119, 0, 170], "isthing": 1, "id": 45, "name": "45"},
     {'color': [171, 134, 1], 'isthing': 1, 'id': 46, 'name': '46'},
     {'color': [109, 63, 54], 'isthing': 1, 'id': 47, 'name': '47'},
     {'color': [207, 138, 255], 'isthing': 1, 'id': 48, 'name': '48'},
     {'color': [151, 0, 95], 'isthing': 1, 'id': 49, 'name': '49'},
     {'color': [9, 80, 61], 'isthing': 1, 'id': 50, 'name': '50'},
     {'color': [84, 105, 51], 'isthing': 1, 'id': 51, 'name': '51'},
     {'color': [74, 65, 105], 'isthing': 1, 'id': 52, 'name': '52'},
     {'color': [166, 196, 102], 'isthing': 1, 'id': 53, 'name': '53'},
     {'color': [208, 195, 210], 'isthing': 1, 'id': 54, 'name': '54'},
     {'color': [255, 109, 65], 'isthing': 1, 'id': 55, 'name': '55'},
     {'color': [0, 143, 149], 'isthing': 1, 'id': 56, 'name': '56'},
     {'color': [179, 0, 194], 'isthing': 1, 'id': 57, 'name': '57'},
     {'color': [209, 99, 106], 'isthing': 1, 'id': 58, 'name': '58'},
     {'color': [5, 121, 0], 'isthing': 1, 'id': 59, 'name': '59'},
     {'color': [227, 255, 205], 'isthing': 1, 'id': 60, 'name': '60'},
]


KINS_CATEGORIES = [
     {'color': [220, 20, 60], 'supercategory': "Living thing", 'id': 1, 'name': 'cyclist'},
     {'color': [119, 11, 32], 'supercategory': "Living thing", 'id': 2, 'name': 'pedestrian'},
     # {'color': [133, 129, 255], 'supercategory': "Living thing", 'id': 3, 'name': 'rider'},
     {'color': [151, 0, 95], 'supercategory': "vehicles", 'id': 4, 'name': 'car'},
     {'color': [0, 0, 230], 'supercategory': "vehicles", 'id': 5, 'name': 'tram'},
     {'color': [106, 0, 228], 'supercategory': "vehicles", 'id': 6, 'name': 'truck'},
     {'color': [0, 60, 100], 'supercategory': "vehicles", 'id': 7, 'name': 'van'},
     {'color': [227, 255, 205], 'supercategory': "vehicles", 'id': 8, 'name': 'misc'},
]


SAILVOS_CATEGORIES = [
    {'name': 'person', 'supercategory': 'person', 'id': 1, 'color': [0, 0, 0]},
    {'name': 'bicycle', 'supercategory': 'bicycle', 'id': 2, 'color': [51, 0, 0]},
    {'name': 'car', 'supercategory': 'car', 'id': 3, 'color': [102, 0, 0]},
    {'name': 'motorcycle', 'supercategory': 'motorcycle', 'id': 4, 'color': [153, 0, 0]},
    {'name': 'airplane', 'supercategory': 'airplane', 'id': 5, 'color': [204, 0, 0]},
    {'name': 'bus', 'supercategory': 'bus', 'id': 6, 'color': [255, 0, 0]},
    {'name': 'train', 'supercategory': 'train', 'id': 7, 'color': [0, 51, 0]},
    {'name': 'truck', 'supercategory': 'truck', 'id': 8, 'color': [51, 51, 0]},
    {'name': 'boat', 'supercategory': 'boat', 'id': 9, 'color': [102, 51, 0]},
    {'name': 'traffic light', 'supercategory': 'traffic light', 'id': 10, 'color': [153, 51, 0]},
    {'name': 'fire hydrant', 'supercategory': 'fire hydrant', 'id': 11, 'color': [204, 51, 0]},
    {'name': 'stop sign', 'supercategory': 'stop sign', 'id': 13, 'color': [255, 51, 0]},
    {'name': 'parking meter', 'supercategory': 'parking meter', 'id': 14, 'color': [0, 102, 0]},
    {'name': 'bench', 'supercategory': 'bench', 'id': 15, 'color': [51, 102, 0]},
    {'name': 'bird', 'supercategory': 'bird', 'id': 16, 'color': [102, 102, 0]},
    {'name': 'cat', 'supercategory': 'cat', 'id': 17, 'color': [153, 102, 0]},
    {'name': 'dog', 'supercategory': 'dog', 'id': 18, 'color': [204, 102, 0]},
    {'name': 'backpack', 'supercategory': 'backpack', 'id': 27, 'color': [255, 102, 0]},
    {'name': 'umbrella', 'supercategory': 'umbrella', 'id': 28, 'color': [0, 153, 0]},
    {'name': 'handbag', 'supercategory': 'handbag', 'id': 31, 'color': [51, 153, 0]},
    {'name': 'tie', 'supercategory': 'tie', 'id': 32, 'color': [102, 153, 0]},
    {'name': 'suitcase', 'supercategory': 'suitcase', 'id': 33, 'color': [153, 153, 0]},
    {'name': 'sports ball', 'supercategory': 'sports ball', 'id': 37, 'color': [204, 153, 0]},
    {'name': 'surfboard', 'supercategory': 'surfboard', 'id': 42, 'color': [255, 153, 0]},
    {'name': 'tennis racket', 'supercategory': 'tennis racket', 'id': 43, 'color': [0, 204, 0]},
    {'name': 'bottle', 'supercategory': 'bottle', 'id': 44, 'color': [51, 204, 0]},
    {'name': 'wine glass', 'supercategory': 'wine glass', 'id': 46, 'color': [102, 204, 0]},
    {'name': 'cup', 'supercategory': 'cup', 'id': 47, 'color': [153, 204, 0]},
    {'name': 'fork', 'supercategory': 'fork', 'id': 48, 'color': [204, 204, 0]},
    {'name': 'knife', 'supercategory': 'knife', 'id': 49, 'color': [255, 204, 0]},
    {'name': 'bowl', 'supercategory': 'bowl', 'id': 51, 'color': [0, 255, 0]},
    {'name': 'banana', 'supercategory': 'banana', 'id': 52, 'color': [51, 255, 0]},
    {'name': 'sandwich', 'supercategory': 'sandwich', 'id': 54, 'color': [102, 255, 0]},
    {'name': 'orange', 'supercategory': 'orange', 'id': 55, 'color': [153, 255, 0]},
    {'name': 'hot dog', 'supercategory': 'hot dog', 'id': 58, 'color': [204, 255, 0]},
    {'name': 'donut', 'supercategory': 'donut', 'id': 60, 'color': [255, 255, 0]},
    {'name': 'chair', 'supercategory': 'chair', 'id': 62, 'color': [0, 0, 51]},
    {'name': 'couch', 'supercategory': 'couch', 'id': 63, 'color': [51, 0, 51]},
    {'name': 'potted plant', 'supercategory': 'potted plant', 'id': 64, 'color': [102, 0, 51]},
    {'name': 'bed', 'supercategory': 'bed', 'id': 65, 'color': [153, 0, 51]},
    {'name': 'dining table', 'supercategory': 'dining table', 'id': 67, 'color': [204, 0, 51]},
    {'name': 'toilet', 'supercategory': 'toilet', 'id': 70, 'color': [255, 0, 51]},
    {'name': 'tv', 'supercategory': 'tv', 'id': 72, 'color': [0, 51, 51]},
    {'name': 'laptop', 'supercategory': 'laptop', 'id': 73, 'color': [51, 51, 51]},
    {'name': 'mouse', 'supercategory': 'mouse', 'id': 74, 'color': [102, 51, 51]},
    {'name': 'remote', 'supercategory': 'remote', 'id': 75, 'color': [153, 51, 51]},
    {'name': 'keyboard', 'supercategory': 'keyboard', 'id': 76, 'color': [204, 51, 51]},
    {'name': 'cell phone', 'supercategory': 'cell phone', 'id': 77, 'color': [255, 51, 51]},
    {'name': 'microwave', 'supercategory': 'microwave', 'id': 78, 'color': [0, 102, 51]},
    {'name': 'toaster', 'supercategory': 'toaster', 'id': 80, 'color': [51, 102, 51]},
    {'name': 'refrigerator', 'supercategory': 'refrigerator', 'id': 82, 'color': [102, 102, 51]},
    {'name': 'book', 'supercategory': 'book', 'id': 84, 'color': [153, 102, 51]},
    {'name': 'clock', 'supercategory': 'clock', 'id': 85, 'color': [204, 102, 51]},
    {'name': 'vase', 'supercategory': 'vase', 'id': 86, 'color': [255, 102, 51]},
    {'name': 'scissors', 'supercategory': 'scissors', 'id': 87, 'color': [0, 153, 51]},
    {'name': 'toothbrush', 'supercategory': 'toothbrush', 'id': 90, 'color': [51, 153, 51]},
    {'name': 'accessories', 'supercategory': 'accessories', 'id': 91, 'color': [102, 153, 51]},
    {'name': 'air conditioner', 'supercategory': 'air conditioner', 'id': 92, 'color': [153, 153, 51]},
    {'name': 'air dancer', 'supercategory': 'air dancer', 'id': 93, 'color': [204, 153, 51]},
    {'name': 'aircraft', 'supercategory': 'aircraft', 'id': 94, 'color': [255, 153, 51]},
    {'name': 'bag', 'supercategory': 'bag', 'id': 97, 'color': [0, 204, 51]},
    {'name': 'ball arcade', 'supercategory': 'ball arcade', 'id': 98, 'color': [51, 204, 51]},
    {'name': 'bbq', 'supercategory': 'bbq', 'id': 100, 'color': [102, 204, 51]},
    {'name': 'bin', 'supercategory': 'bin', 'id': 104, 'color': [153, 204, 51]},
    {'name': 'blood', 'supercategory': 'blood', 'id': 106, 'color': [204, 204, 51]},
    {'name': 'board', 'supercategory': 'board', 'id': 107, 'color': [255, 204, 51]},
    {'name': 'box', 'supercategory': 'box', 'id': 112, 'color': [0, 255, 51]},
    {'name': 'bread', 'supercategory': 'bread', 'id': 113, 'color': [51, 255, 51]},
    {'name': 'bus wreck', 'supercategory': 'bus wreck', 'id': 115, 'color': [102, 255, 51]},
    {'name': 'cabinet', 'supercategory': 'cabinet', 'id': 116, 'color': [153, 255, 51]},
    {'name': 'cactus', 'supercategory': 'cactus', 'id': 117, 'color': [204, 255, 51]},
    {'name': 'calculator', 'supercategory': 'calculator', 'id': 118, 'color': [255, 255, 51]},
    {'name': 'camera', 'supercategory': 'camera', 'id': 119, 'color': [0, 0, 102]},
    {'name': 'can', 'supercategory': 'can', 'id': 120, 'color': [51, 0, 102]},
    {'name': 'candle', 'supercategory': 'candle', 'id': 121, 'color': [102, 0, 102]},
    {'name': 'canopy', 'supercategory': 'canopy', 'id': 122, 'color': [153, 0, 102]},
    {'name': 'car wreck', 'supercategory': 'car wreck', 'id': 124, 'color': [204, 0, 102]},
    {'name': 'chime', 'supercategory': 'chime', 'id': 128, 'color': [255, 0, 102]},
    {'name': 'cigarette', 'supercategory': 'cigarette', 'id': 129, 'color': [0, 51, 102]},
    {'name': 'cloth', 'supercategory': 'cloth', 'id': 131, 'color': [51, 51, 102]},
    {'name': 'crate', 'supercategory': 'crate', 'id': 133, 'color': [102, 51, 102]},
    {'name': 'door', 'supercategory': 'door', 'id': 138, 'color': [153, 51, 102]},
    {'name': 'drug burner', 'supercategory': 'drug burner', 'id': 139, 'color': [204, 51, 102]},
    {'name': 'dvd', 'supercategory': 'dvd', 'id': 140, 'color': [255, 51, 102]},
    {'name': 'fan', 'supercategory': 'fan', 'id': 141, 'color': [0, 102, 102]},
    {'name': 'fax', 'supercategory': 'fax', 'id': 142, 'color': [51, 102, 102]},
    {'name': 'fence', 'supercategory': 'fence', 'id': 143, 'color': [102, 102, 102]},
    {'name': 'flag', 'supercategory': 'flag', 'id': 145, 'color': [153, 102, 102]},
    {'name': 'flag hook', 'supercategory': 'flag hook', 'id': 146, 'color': [204, 102, 102]},
    {'name': 'frame', 'supercategory': 'frame', 'id': 148, 'color': [255, 102, 102]},
    {'name': 'fringe', 'supercategory': 'fringe', 'id': 149, 'color': [0, 153, 102]},
    {'name': 'gas pump', 'supercategory': 'gas pump', 'id': 150, 'color': [51, 153, 102]},
    {'name': 'gas tank', 'supercategory': 'gas tank', 'id': 151, 'color': [102, 153, 102]},
    {'name': 'generator', 'supercategory': 'generator', 'id': 152, 'color': [153, 153, 102]},
    {'name': 'glass stack', 'supercategory': 'glass stack', 'id': 153, 'color': [204, 153, 102]},
    {'name': 'glasses', 'supercategory': 'glasses', 'id': 154, 'color': [255, 153, 102]},
    {'name': 'glasses stand', 'supercategory': 'glasses stand', 'id': 155, 'color': [0, 204, 102]},
    {'name': 'gorilla', 'supercategory': 'gorilla', 'id': 156, 'color': [51, 204, 102]},
    {'name': 'grenade', 'supercategory': 'grenade', 'id': 157, 'color': [102, 204, 102]},
    {'name': 'group of stuffs', 'supercategory': 'group of stuffs', 'id': 158, 'color': [153, 204, 102]},
    {'name': 'guitar', 'supercategory': 'guitar', 'id': 159, 'color': [204, 204, 102]},
    {'name': 'gun', 'supercategory': 'gun', 'id': 160, 'color': [255, 204, 102]},
    {'name': 'gun mag', 'supercategory': 'gun mag', 'id': 161, 'color': [0, 255, 102]},
    {'name': 'headset', 'supercategory': 'headset', 'id': 163, 'color': [51, 255, 102]},
    {'name': 'heavy duty car', 'supercategory': 'heavy duty car', 'id': 164, 'color': [102, 255, 102]},
    {'name': 'helicopter', 'supercategory': 'helicopter', 'id': 165, 'color': [153, 255, 102]},
    {'name': 'holster', 'supercategory': 'holster', 'id': 166, 'color': [204, 255, 102]},
    {'name': 'ink', 'supercategory': 'ink', 'id': 168, 'color': [255, 255, 102]},
    {'name': 'insect', 'supercategory': 'insect', 'id': 169, 'color': [0, 0, 153]},
    {'name': 'iron', 'supercategory': 'iron', 'id': 170, 'color': [51, 0, 153]},
    {'name': 'ladder', 'supercategory': 'ladder', 'id': 173, 'color': [102, 0, 153]},
    {'name': 'lamp', 'supercategory': 'lamp', 'id': 174, 'color': [153, 0, 153]},
    {'name': 'letter box', 'supercategory': 'letter box', 'id': 176, 'color': [204, 0, 153]},
    {'name': 'light', 'supercategory': 'light', 'id': 177, 'color': [255, 0, 153]},
    {'name': 'machanical device', 'supercategory': 'machanical device', 'id': 178, 'color': [0, 51, 153]},
    {'name': 'magnet', 'supercategory': 'magnet', 'id': 179, 'color': [51, 51, 153]},
    {'name': 'mat', 'supercategory': 'mat', 'id': 180, 'color': [102, 51, 153]},
    {'name': 'microphone', 'supercategory': 'microphone', 'id': 181, 'color': [153, 51, 153]},
    {'name': 'monster', 'supercategory': 'monster', 'id': 183, 'color': [204, 51, 153]},
    {'name': 'mushroom', 'supercategory': 'mushroom', 'id': 186, 'color': [255, 51, 153]},
    {'name': 'nail file', 'supercategory': 'nail file', 'id': 187, 'color': [0, 102, 153]},
    {'name': 'ngcan', 'supercategory': 'ngcan', 'id': 188, 'color': [51, 102, 153]},
    {'name': 'nwcan', 'supercategory': 'nwcan', 'id': 189, 'color': [102, 102, 153]},
    {'name': 'pallet pile', 'supercategory': 'pallet pile', 'id': 191, 'color': [153, 102, 153]},
    {'name': 'paper', 'supercategory': 'paper', 'id': 192, 'color': [204, 102, 153]},
    {'name': 'parachute', 'supercategory': 'parachute', 'id': 193, 'color': [255, 102, 153]},
    {'name': 'phone box', 'supercategory': 'phone box', 'id': 196, 'color': [0, 153, 153]},
    {'name': 'pig', 'supercategory': 'pig', 'id': 197, 'color': [51, 153, 153]},
    {'name': 'pillar', 'supercategory': 'pillar', 'id': 198, 'color': [102, 153, 153]},
    {'name': 'pillow', 'supercategory': 'pillow', 'id': 199, 'color': [153, 153, 153]},
    {'name': 'pipe', 'supercategory': 'pipe', 'id': 200, 'color': [204, 153, 153]},
    {'name': 'plate', 'supercategory': 'plate', 'id': 201, 'color': [255, 153, 153]},
    {'name': 'portaloo', 'supercategory': 'portaloo', 'id': 202, 'color': [0, 204, 153]},
    {'name': 'pot', 'supercategory': 'pot', 'id': 203, 'color': [51, 204, 153]},
    {'name': 'printer', 'supercategory': 'printer', 'id': 205, 'color': [102, 204, 153]},
    {'name': 'projectile', 'supercategory': 'projectile', 'id': 206, 'color': [153, 204, 153]},
    {'name': 'pumpkin', 'supercategory': 'pumpkin', 'id': 207, 'color': [204, 204, 153]},
    {'name': 'rabbit', 'supercategory': 'rabbit', 'id': 208, 'color': [255, 204, 153]},
    {'name': 'rack', 'supercategory': 'rack', 'id': 209, 'color': [0, 255, 153]},
    {'name': 'rail wreck', 'supercategory': 'rail wreck', 'id': 210, 'color': [51, 255, 153]},
    {'name': 'ramp', 'supercategory': 'ramp', 'id': 211, 'color': [102, 255, 153]},
    {'name': 'rat', 'supercategory': 'rat', 'id': 212, 'color': [153, 255, 153]},
    {'name': 'road barrier', 'supercategory': 'road barrier', 'id': 215, 'color': [204, 255, 153]},
    {'name': 'roll', 'supercategory': 'roll', 'id': 216, 'color': [255, 255, 153]},
    {'name': 'roller car', 'supercategory': 'roller car', 'id': 217, 'color': [0, 0, 204]},
    {'name': 'roofvent', 'supercategory': 'roofvent', 'id': 218, 'color': [51, 0, 204]},
    {'name': 'rope', 'supercategory': 'rope', 'id': 219, 'color': [102, 0, 204]},
    {'name': 'rose', 'supercategory': 'rose', 'id': 220, 'color': [153, 0, 204]},
    {'name': 'rubwee', 'supercategory': 'rubwee', 'id': 221, 'color': [204, 0, 204]},
    {'name': 'satdish', 'supercategory': 'satdish', 'id': 223, 'color': [255, 0, 204]},
    {'name': 'scrap', 'supercategory': 'scrap', 'id': 225, 'color': [0, 51, 204]},
    {'name': 'shelter', 'supercategory': 'shelter', 'id': 226, 'color': [51, 51, 204]},
    {'name': 'shoe', 'supercategory': 'shoe', 'id': 227, 'color': [102, 51, 204]},
    {'name': 'smoke alarm', 'supercategory': 'smoke alarm', 'id': 228, 'color': [153, 51, 204]},
    {'name': 'speaker', 'supercategory': 'speaker', 'id': 229, 'color': [204, 51, 204]},
    {'name': 'stair', 'supercategory': 'stair', 'id': 231, 'color': [255, 51, 204]},
    {'name': 'stape', 'supercategory': 'stape', 'id': 232, 'color': [0, 102, 204]},
    {'name': 'statue', 'supercategory': 'statue', 'id': 233, 'color': [51, 102, 204]},
    {'name': 'stick', 'supercategory': 'stick', 'id': 234, 'color': [102, 102, 204]},
    {'name': 'stone', 'supercategory': 'stone', 'id': 235, 'color': [153, 102, 204]},
    {'name': 'street stand', 'supercategory': 'street stand', 'id': 237, 'color': [204, 102, 204]},
    {'name': 'tape player', 'supercategory': 'tape player', 'id': 240, 'color': [255, 102, 204]},
    {'name': 'telescope', 'supercategory': 'telescope', 'id': 241, 'color': [0, 153, 204]},
    {'name': 'tennis net', 'supercategory': 'tennis net', 'id': 242, 'color': [51, 153, 204]},
    {'name': 'tent', 'supercategory': 'tent', 'id': 244, 'color': [102, 153, 204]},
    {'name': 'tire', 'supercategory': 'tire', 'id': 246, 'color': [153, 153, 204]},
    {'name': 'tool', 'supercategory': 'tool', 'id': 249, 'color': [204, 153, 204]},
    {'name': 'torture', 'supercategory': 'torture', 'id': 251, 'color': [255, 153, 204]},
    {'name': 'tray', 'supercategory': 'tray', 'id': 254, 'color': [0, 204, 204]},
    {'name': 'tree', 'supercategory': 'tree', 'id': 255, 'color': [51, 204, 204]},
    {'name': 'trimmer', 'supercategory': 'trimmer', 'id': 256, 'color': [102, 204, 204]},
    {'name': 'trolley', 'supercategory': 'trolley', 'id': 257, 'color': [153, 204, 204]},
    {'name': 'truck trailer', 'supercategory': 'truck trailer', 'id': 259, 'color': [204, 204, 204]},
    {'name': 'tube', 'supercategory': 'tube', 'id': 260, 'color': [255, 204, 204]},
    {'name': 'valve', 'supercategory': 'valve', 'id': 263, 'color': [0, 255, 204]},
    {'name': 'vending machine', 'supercategory': 'vending machine', 'id': 265, 'color': [51, 255, 204]},
    {'name': 'volt meter', 'supercategory': 'volt meter', 'id': 266, 'color': [102, 255, 204]},
    {'name': 'wall', 'supercategory': 'wall', 'id': 267, 'color': [153, 255, 204]},
    {'name': 'water wheel', 'supercategory': 'water wheel', 'id': 268, 'color': [204, 255, 204]},
    {'name': 'wheel', 'supercategory': 'wheel', 'id': 269, 'color': [255, 255, 204]},
    {'name': 'wheel barrow', 'supercategory': 'wheel barrow', 'id': 270, 'color': [0, 0, 255]},
    {'name': 'window', 'supercategory': 'window', 'id': 271, 'color': [51, 0, 255]},
    {'name': 'wood', 'supercategory': 'wood', 'id': 273, 'color': [102, 0, 255]}]


sailvos_ignore = [7, 27, 48, 54, 55, 58, 70, 78, 106, 128, 153, 187, 193, 207, 221, 232, 240, 254, 266, 268]


def _get_coco_instances_meta():
    thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 80, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_coco_amodal_instances_meta():
    ret = {'thing_colors': [[220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228], [0, 60, 100], [0, 80, 100],
                      [0, 0, 70], [0, 0, 192], [250, 170, 30], [100, 170, 30], [220, 220, 0], [175, 116, 175],
                      [250, 0, 30], [165, 42, 42], [255, 77, 255], [0, 226, 252], [182, 182, 255], [0, 82, 0],
                      [120, 166, 157], [110, 76, 0], [174, 57, 255], [199, 100, 0], [72, 0, 118], [255, 179, 240],
                      [0, 125, 92], [209, 0, 151], [188, 208, 182], [0, 220, 176], [255, 99, 164], [92, 0, 73],
                      [133, 129, 255], [78, 180, 255], [0, 228, 0], [174, 255, 243], [45, 89, 255], [134, 134, 103],
                      [145, 148, 174], [255, 208, 186], [197, 226, 255], [171, 134, 1], [109, 63, 54], [207, 138, 255],
                      [151, 0, 95], [9, 80, 61], [84, 105, 51], [74, 65, 105], [166, 196, 102], [208, 195, 210],
                      [255, 109, 65], [0, 143, 149], [179, 0, 194], [209, 99, 106], [5, 121, 0], [227, 255, 205],
                      [147, 186, 208], [153, 69, 1], [3, 95, 161], [163, 255, 0], [119, 0, 170], [0, 182, 199],
                      [0, 165, 120], [183, 130, 88], [95, 32, 0], [130, 114, 135], [110, 129, 133], [166, 74, 118],
                      [219, 142, 185], [79, 210, 114], [178, 90, 62], [65, 70, 15], [127, 167, 115], [59, 105, 106],
                      [142, 108, 45], [196, 172, 0], [95, 54, 80], [128, 76, 255], [201, 57, 1], [246, 0, 122],
                      [191, 162, 208]],
           "thing_classes": ["object"]}

    return ret


def _get_coco_amodal_cls_instances_meta():
    thing_ids = [k["id"] for k in COCOA_CLS_CATEGORIES]
    thing_colors = [k["color"] for k in COCOA_CLS_CATEGORIES]
    assert len(thing_ids) == 80, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCOA_CLS_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_coco_panoptic_separated_meta():
    """
    Returns metadata for "separated" version of the panoptic segmentation dataset.
    """
    stuff_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 0]
    assert len(stuff_ids) == 53, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 53], used in models) to ids in the dataset (used for processing results)
    # The id 0 is mapped to an extra category "thing".
    stuff_dataset_id_to_contiguous_id = {k: i + 1 for i, k in enumerate(stuff_ids)}
    # When converting COCO panoptic annotations to semantic annotations
    # We label the "thing" category to 0
    stuff_dataset_id_to_contiguous_id[0] = 0

    # 54 names for COCO stuff categories (including "things")
    stuff_classes = ["things"] + [
        k["name"].replace("-other", "").replace("-merged", "")
        for k in COCO_CATEGORIES
        if k["isthing"] == 0
    ]

    # NOTE: I randomly picked a color for things
    stuff_colors = [[82, 18, 128]] + [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 0]
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    ret.update(_get_coco_instances_meta())
    return ret


def _get_d2sa_instances_meta():
    thing_ids = [k["id"] for k in D2SA_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in D2SA_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 60, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in D2SA_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_kins_instances_meta():
    thing_ids = [k["id"] for k in KINS_CATEGORIES]
    thing_colors = [k["color"] for k in KINS_CATEGORIES]
    assert len(thing_ids) == 7, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in KINS_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_sailvos_instances_meta():
    SAILVOS_CATEGORIES_ = [c for c in SAILVOS_CATEGORIES if c['id'] not in sailvos_ignore]
    # SAILVOS_CATEGORIES_ = SAILVOS_CATEGORIES
    thing_ids = [k["id"] for k in SAILVOS_CATEGORIES_]
    thing_colors = [k["color"] for k in SAILVOS_CATEGORIES_]
    assert len(thing_ids) == 163, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SAILVOS_CATEGORIES_]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
        "ignore_classes": sailvos_ignore
    }
    return ret


def _get_builtin_metadata(dataset_name):
    if dataset_name == "coco":
        return _get_coco_instances_meta()
    if dataset_name == 'cocoa':
        return _get_coco_amodal_instances_meta()
    if dataset_name == 'coco_amodal_cls':
        return _get_coco_amodal_cls_instances_meta()
    if dataset_name == "coco_panoptic_separated":
        return _get_coco_panoptic_separated_meta()
    elif dataset_name == "coco_person":
        return {
            "thing_classes": ["person"],
            "keypoint_names": COCO_PERSON_KEYPOINT_NAMES,
            "keypoint_flip_map": COCO_PERSON_KEYPOINT_FLIP_MAP,
            "keypoint_connection_rules": KEYPOINT_CONNECTION_RULES,
        }
    elif dataset_name == "cityscapes":
        # fmt: off
        CITYSCAPES_THING_CLASSES = [
            "person", "rider", "car", "truck",
            "bus", "train", "motorcycle", "bicycle",
        ]
        CITYSCAPES_STUFF_CLASSES = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
            "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle", "license plate",
        ]
        # fmt: on
        return {
            "thing_classes": CITYSCAPES_THING_CLASSES,
            "stuff_classes": CITYSCAPES_STUFF_CLASSES,
        }
    elif dataset_name == "d2sa":
        return _get_d2sa_instances_meta()
    elif dataset_name == "kins":
        return _get_kins_instances_meta()
    elif dataset_name == "sailvos":
        return _get_sailvos_instances_meta()
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
