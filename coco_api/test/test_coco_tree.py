# (c) EPFL - LTS5, 2023

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

import dataclasses
import json
import os

from coco_api import COCOAnnotations


def run_test(test_dir_name: str):
	annot_fname = "annotations.json"
	os.chdir(os.path.join(Path(__file__).parent, __name__, test_dir_name))
	with open(annot_fname) as fp:
		annot_dict = json.load(fp)
	coco_annot = COCOAnnotations(**annot_dict)
	#coco_annot_dict = dataclasses.asdict(coco_annot)
	#assert coco_annot_dict == annot_dict

def test_coco_image_all():
	run_test("test_coco_image_all")

def test_coco_image_required():
	run_test("test_coco_image_required")
