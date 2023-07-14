# (c) EPFL - LTS5, 2022

from dataclasses import dataclass
from typing import Optional

from .utils import indexing_tree_expander


@dataclass(slots=True)
class COCOImage:
	id: int
	width: int
	height: int
	file_name: str
	license: Optional[str] = None
	flickr_url: Optional[str] = None
	coco_url: Optional[str] = None
	date_captured: Optional[str] = None
	flickr_640_url: Optional[str] = None


@dataclass(slots=True)
class COCOCategory:
	supercategory: str
	id: int
	name: str


@dataclass(slots=True)
class COCOAnnotation:
	id: int
	image_id: int
	category_id: int
	segmentation: list[list[int]]
	area: float
	bbox: list[int]
	iscrowd: int


@dataclass(slots=True)
class COCOInfo:
	year: int
	version: str
	description: str
	contributor: str
	url: str
	date_created: str


@dataclass(slots=True)
@indexing_tree_expander(index_key="id")
class COCOAnnotations:
	images: dict[int, COCOImage]
	categories: dict[int, COCOCategory]
	annotations: dict[int, COCOAnnotation]
	licenses: list
	info: COCOInfo
