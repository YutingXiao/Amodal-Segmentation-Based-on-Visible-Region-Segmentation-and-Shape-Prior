# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import io
import logging
import contextlib
import os
import datetime
import json
import numpy as np
import imagesize

from PIL import Image

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode, PolygonMasks, Boxes
from fvcore.common.file_io import PathManager


from .. import MetadataCatalog, DatasetCatalog

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_sailvos_json"]

useful_cats = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 28, 31, 32, 33, 37, 42, 43, 44, 46, 47, 49, 51,
               52, 60, 62, 63, 64, 65, 67, 72, 73, 74, 75, 76, 77, 80, 82, 84, 85, 86, 87, 90, 91, 92, 93, 94, 97, 98,
               100, 104, 107, 112, 113, 115, 116, 117, 118, 119, 120, 121, 122, 124, 129, 131, 133, 138, 139, 140, 141,
               142, 143, 145, 146, 148, 149, 150, 151, 152, 154, 155, 156, 157, 158, 159, 160, 161, 163, 164, 165, 166,
               168, 169, 170, 173, 174, 176, 177, 178, 179, 180, 181, 183, 186, 188, 189, 191, 192, 196, 197, 198, 199,
               200, 201, 202, 203, 205, 206, 208, 209, 210, 211, 212, 215, 216, 217, 218, 219, 220, 223, 225, 226, 227,
               228, 229, 231, 233, 234, 235, 237, 241, 242, 244, 246, 249, 251, 255, 256, 257, 259, 260, 263, 265, 267,
               269, 270, 271, 273]


def load_sailvos_json(json_file, image_root, dataset_name=None):
    """
    Load a json file with D2SA's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in D2SA instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        sailvos_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))
    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        all_cat_ids = sorted(sailvos_api.getCatIds())
        cat_ids = [i for i in all_cat_ids if i not in meta.ignore_classes]
        cats = sailvos_api.loadCats(cat_ids)
        thing_classes = [str(c["name"]) for c in cats]
        meta.thing_classes = thing_classes

        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
            )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(list(sailvos_api.imgs.keys()))
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = sailvos_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [sailvos_api.imgToAnns[img_id] for img_id in img_ids]

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "category_id"]

    num_instances_without_valid_segmentation = 0
    num_instances_without_valid_visible_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]
        objs = []
        for anno in anno_dict_list:
            if anno['occlude_rate'] > 0.95:
                continue
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}
            segm = anno.get("visible_mask", None) if dataset_name.endswith("visible") else anno.get("segmentation", None)
            # segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if not isinstance(segm, dict):
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

                vis_segm = anno.get("visible_mask", None)
                if not isinstance(vis_segm, dict):
                    # filter out invalid polygons (< 3 points)
                    vis_segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(vis_segm) == 0:
                        num_instances_without_valid_visible_segmentation += 1
                        continue  # ignore this instance
                obj["visible_mask"] = vis_segm
                obj["occlude_rate"] = anno.get("occlude_rate", 0)

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        if len(objs) == 0:
            continue
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warn(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )

    if num_instances_without_valid_visible_segmentation > 0:
        logger.warn(
            "Filtered out {} instances without valid visible segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_visible_segmentation
            )
        )
    return dataset_dicts


if __name__ == "__main__":
    """
    Test the d2sa json dataset loader.

    Usage:
        python -m detectron2.data.datasets.d2sa \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "d2sa_val", or other
        pre-registered ones
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys

    logger = setup_logger(name=__name__)
    assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = load_sailvos_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "d2sa-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
