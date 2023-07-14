# (c) EPFL - LTS5, 2022

import json
import os
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path
from typing import Generic, Iterable, TypeVar

from PIL import Image, ImageDraw, ImageFont
from pqdm.processes import pqdm

from coco_api import COCOAnnotations
from coco_api.coco_tree import COCOAnnotation, COCOImage

TAnnotations = TypeVar("TAnnotations", bound=COCOAnnotations)


class ValidationError(Exception):
	def __init__(self, type: str, file_name: str, message: str):
		super().__init__(f"Validation failed for {type} '{file_name}': {message}")

class FileValidationError(ValidationError):
	def __init__(self, file_name: str, message: str):
		super().__init__("file", file_name, message)

class ImageValidationError(ValidationError):
	def __init__(self, file_name: str, message: str):
		super().__init__("image", file_name, message)


class PlotType(Enum):
	ANNOTATION = auto()
	MASK = auto()


class COCOHandler(Generic[TAnnotations]):
	@staticmethod
	def _get_img_name(file_name: str) -> str:
		return Path(file_name).stem
	
	@staticmethod
	def _open_image(file_name: str, img_root_dir: str) -> Image.Image:
		img = Image.open(os.path.join(img_root_dir, file_name))
		return img
	
	_coco_annots: TAnnotations
	_img_name_img_id_dict: dict[str, int]
	_img_id_annots_ids_dict: dict[int, list[int]]
	
	@property
	def annotations(self) -> TAnnotations:
		return self._coco_annots
	
	@property
	def img_name_list(self) -> list[str]:
		return list(self._img_name_img_id_dict)
	
	def __init__(self, annot_path: str, tree_class: type[TAnnotations] = COCOAnnotations):
		with open(annot_path) as fp:
			json_dict = json.load(fp)
		self._coco_annots: TAnnotations = tree_class(**json_dict)
		self._img_name_img_id_dict = {}
		for img_id, coco_img in self._coco_annots.images.items():
			img_name = COCOHandler._get_img_name(coco_img.file_name)
			self._img_name_img_id_dict[img_name] = coco_img.id
		self._img_id_annots_ids_dict = defaultdict(list)
		for annot in self._coco_annots.annotations.values():
			self._img_id_annots_ids_dict[annot.image_id].append(annot.id)
		# TODO validate used categories against defined categories?
	
	def get_img_id(self, img_name: str) -> int:
		return self._img_name_img_id_dict[img_name]
	
	def get_img_metadata(
		self,
		*,
		file_name: str | None = None,
		img_name: str | None = None,
		img_id: int | None = None
	) -> COCOImage:
		assert sum(attr is not None for attr in (file_name, img_name, img_id)) == 1, \
			"One and only one parameter among 'file_name', 'img_name' and 'img_id' should be specified"
		if img_id is None:
			if img_name is None:
				img_name = self._get_img_name(file_name)
			img_id = self.get_img_id(img_name)
		img_metadata = self._coco_annots.images[img_id]
		return img_metadata
	
	def get_img_annots(
			self,
			*,
			file_name: str | None = None,
			img_name: str | None = None,
			img_id: int | None = None
		) -> list[COCOAnnotation]:
		assert sum(attr is not None for attr in (file_name, img_name, img_id)) == 1, \
			"One and only one parameter among 'file_name', 'img_name' and 'img_id' should be specified"
		if img_id is None:
			if img_name is None:
				img_name = self._get_img_name(file_name)
			img_id = self.get_img_id(img_name)
		annots_ids = self._img_id_annots_ids_dict[img_id]
		annots = [self.annotations.annotations[annots_id] for annots_id in annots_ids]
		return annots
	
	def validate_img_dir(self, img_root_dir: str, ext: str | None = None):
		for coco_img in self._coco_annots.images.values():
			img_name = COCOHandler._get_img_name(coco_img.file_name)
			file_name = coco_img.file_name if ext is None else f"{img_name}.{ext}"
			try:
				img: Image.Image = COCOHandler._open_image(file_name, img_root_dir)
			except Exception as ex:
				raise FileValidationError(file_name, f"Could not open image: {ex}")
			else:
				if img.size != (coco_img.width, coco_img.height):
					raise FileValidationError(file_name, "Size of image not matching with metadata")
				img.close()
	
	def validate_img_name_list(self, img_name_list: Iterable[str]):
		for img_name in img_name_list:
			if img_name not in self._img_name_img_id_dict:
				raise ImageValidationError(img_name, "Image name not in annotations")
	
	def load_image(
		self,
		img_root_dir: str,
		*,
		file_name: str | None = None,
		img_name: str | None = None,
		img_id: int | None = None
	) -> Image.Image:
		assert sum(attr is not None for attr in (file_name, img_name, img_id)) == 1, \
			"One and only one parameter among 'file_name', 'img_name' and 'img_id' should be specified"
		if file_name is None:
			file_name = self.get_img_metadata(img_name=img_name, img_id=img_id).file_name
		img = COCOHandler._open_image(file_name, img_root_dir)
		img.load()
		return img
	
	def _draw_annots(self, img: Image.Image, img_annot_list: list[COCOAnnotation]) -> Image.Image:
		img = img.copy()
		draw = ImageDraw.Draw(img, "RGBA")
		for annot in img_annot_list:
			# segmentation area
			assert len(annot.segmentation) == 1
			draw.polygon(annot.segmentation[0], fill="#ff000055", outline="#ff0000", width=4)
			# classification bounding box
			bbox = annot.bbox
			bbox = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
			draw.rectangle(bbox, outline="#ff00ff", width=4)
			# classification text
			text_pos = bbox
			font_size = round(min(img.size) / 60)
			category_name = self._coco_annots.categories[annot.category_id].name
			font = ImageFont.truetype("LiberationSans-Regular.ttf", font_size)
			text_bbox = draw.textbbox(text_pos, category_name, anchor="lb", font=font)
			draw.rectangle(text_bbox, fill="#ff00ff")
			draw.text(text_pos, category_name, anchor="lb", font=font, fill="white")
		return img
	
	def _draw_mask(self, img: Image.Image, img_annot_list: list[COCOAnnotation]) -> Image.Image:
		img = Image.new("1", img.size)
		draw = ImageDraw.Draw(img, "1")
		for annot in img_annot_list:
			# segmentation area
			assert len(annot.segmentation) == 1
			draw.polygon(annot.segmentation[0], fill="#ffffff", outline="#ffffff", width=1)
		return img
	
	def draw(
		self,
		img_name: str,
		img_root_dir: str,
		plot_type: PlotType,
		categories: list[int] | None = None
	) -> Image.Image:
		file_name = self.get_img_metadata(img_name=img_name).file_name
		img_annot_list = self.get_img_annots(img_name=img_name)
		if categories is not None:
			img_annot_list = list(filter(lambda img_annot: img_annot.category_id in categories, img_annot_list))
		with COCOHandler._open_image(file_name, img_root_dir) as img:
			match plot_type:
				case PlotType.ANNOTATION:
					img = self._draw_annots(img, img_annot_list)
				case PlotType.MASK:
					img = self._draw_mask(img, img_annot_list)
		return img
	
	def _plot(
		self,
		img_name: str,
		img_root_dir: str,
		out_dir: str,
		plot_type: PlotType,
		categories: list[int] | None
	):
		img = self.draw(img_name, img_root_dir, plot_type, categories)
		img.save(os.path.join(out_dir, f"{img_name}.png"))
	
	def plot(
		self,
		img_name_list: list[str],
		img_root_dir: str,
		out_dir: str,
		plot_type: PlotType = PlotType.ANNOTATION,
		*,
		categories: list[int] | None = None,
		n_jobs: int = 8
	):
		os.makedirs(out_dir, exist_ok=True)
		pqdm(
			((img_name, img_root_dir, out_dir, plot_type, categories) for img_name in img_name_list),
			self._plot,
			argument_type="args",
			n_jobs=n_jobs,
			exception_behaviour="immediate",
			dynamic_ncols=True)
