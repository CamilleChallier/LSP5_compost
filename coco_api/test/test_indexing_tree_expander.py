# (c) EPFL - LTS5, 2023

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))


import copy
from dataclasses import dataclass, fields
from typing import Optional

from coco_api.utils import indexing_tree_expander


def test_decorator():
	@dataclass
	@indexing_tree_expander
	class ITE_DC:
		flt: float
		lst: list[float]
		dct: dict[int, float]
		opt: Optional[float] = None
	@dataclass
	@indexing_tree_expander()
	class ITE2_DC:
		flt: float
		lst: list[float]
		dct: dict[int, float]
		opt: Optional[float] = None
	init = {
		"flt": 1.,
		"lst": [1.],
		"dct": {0: 1.}
	}
	opt_init = copy.deepcopy(init)
	opt_init["opt"] = 1.
	ITE_DC(**init)
	ITE_DC(**opt_init)
	ITE2_DC(**init)
	ITE2_DC(**opt_init)


def test_indexing_expansion():
	@dataclass
	class Leaf:
		flt: Optional[float]
		lst: list[float]
		dct: dict[int, float]
	@dataclass
	class Mid:
		id: int
		flt: Optional[float]
		lst: list[Leaf]
		dct: dict[int, Leaf]
	@dataclass
	class Root:
		lst: list[Mid]
		dct: dict[int, Mid]
		idx: dict[int, Mid]
		flt: Optional[float] = None
	@dataclass
	@indexing_tree_expander(index_key="id")
	class ITE_Root(Root):
		pass
	leaf = {
		"flt": 3.,
		"lst": [3.],
		"dct": {3: 3.}
	}
	mid = {
		"id": 0,
		"flt": None,
		"lst": [copy.deepcopy(leaf)],
		"dct": {2: copy.deepcopy(leaf)}
	}
	root = {
		"lst": [copy.deepcopy(mid)],
		"dct": {1: copy.deepcopy(mid)},
		"idx": [copy.deepcopy(mid)]
	}
	res_leaf = Leaf(**leaf)
	res_mid = Mid(
		id=0,
		flt=None,
		lst=[copy.deepcopy(res_leaf)],
		dct={2: copy.deepcopy(res_leaf)}
	)
	res_root = Root(
		lst=[copy.deepcopy(res_mid)],
		dct={1: copy.deepcopy(res_mid)},
		idx={0: copy.deepcopy(res_mid)}
	)
	tree = ITE_Root(**root)
	for field in fields(tree):
		tree_field_val = getattr(tree, field.name)
		res_field_val = getattr(res_root, field.name)
		assert tree_field_val == res_field_val
